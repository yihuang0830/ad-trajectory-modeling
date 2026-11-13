"""pretrain.py -- 先在 NACC 上从零预训练，再到 ROSMAP 上微调（2026-10-22 加）。

为什么：ROSMAP 训练的人不多。从零训练的 transformer / GPT 不比站点上的梯度提升好 ——
一个可能的原因是数据太少、能看历史的大模型学不出来。
NACC（nacc_data.py：同一种输入格式、同一套题）大得多：先在 NACC 上训练到收敛，再拿 ROSMAP 的 train 微调。

三种起点（同一份代码、同一组 seed，只差初始化），都在 ROSMAP 的 split811 上：train 训练 / 微调，test 早停、挑学习率，val 只报一次：
  scratch   从随机初始化在 ROSMAP 上训练（对照）
  nacc      NACC 预训练的权重 -> ROSMAP 微调（全部参数一起调；学习率 {原值, 原值 / 3} 在 test 段上挑）
  zeroshot  NACC 预训练的权重直接在 ROSMAP 上读出，不微调（NACC 没有的输入 —— 化验、睡眠、炎症、跛行、饮酒、子研究 —— 读出时改成"没测"，
            和预训练时看到的一样）。只说明 NACC 学到的东西有多少能直接搬过来。
模型：
  seq       sequence.py 的因果 transformer（多任务头 60 个 sigmoid），超参照 sequence.SEQ_HP，5 seed 集成。
            预训练检查点 $ZOO_RUN_DIR/nacc/ckpt/ 里每个 seed 一个，nacc 和 zeroshot 共用。
  seqsim    rollout_versions 的 transformer-sim（风险头 + 下一次访视分布头），能生成未来，单个 seed（和 transformer-sim 一样）。
  ethos     ETHOS：分位数切点和词表在 ROSMAP train 上定，同一份分词同时用于 NACC 和 ROSMAP，单个 seed（和 ethos 一样）。
  seqar     选定的 transformer（VISTA；rollout_versions.SeqARModel，只学下一步：下一年内死亡 / AD + 下一次访视的分布），
            2026-11-10 加，5 个 seed，各自一个预训练检查点；微调见 SeqARNACC（run.py --variant naccft），对照是基因实验的
            transformer-base-s0..s4（同一份数据、同一个训练函数、只在 ROSMAP 上训练）。分析见 vista_nacc_compare.py。
训练循环（train_loop）和 sequence.fit 是同一个算法（AdamW、梯度裁剪 1.0、按 test 段损失早停、取最好的 epoch），只是每个 batch 截到这批里最长的那条记录（填充的位置本来就被遮住，数学上一样，只是快）。
NACC 预训练另外按记录长度分桶组 batch（NACC 很多人只有几次访视），每个 epoch 存一次可续训的检查点：RIS 的短分区只给 30 分钟，
超时了下一个作业接着训；同一个 seed 有锁文件，两个作业不会同时训。
"""
import json
import os
import time

import numpy as np
import torch

import bl_data as B
from bl_data import I_MC, I_MS, I_XS
from common import RUN_DIR

NACC_DIR = os.path.join(RUN_DIR, "nacc")
CKPT = os.environ.get("NACC_CKPT_DIR") or os.path.join(NACC_DIR, "ckpt")      # 冒烟测试指到别的目录，不碰正式检查点
# NACC 里没有的输入（nacc_data.py）：连续量 5..14（血糖、HbA1c、HDL、LDL、eGFR、PSQI、Berlin、CRP、IL-6、TNF-α）、状态 3（跛行）
NACC_MISSING_CONT = list(range(5, 15))
NACC_MISSING_STATE = [3]
S_ALC, S_ALC_NA, S_MAP, S_LATC, S_PREV_CLAUD = 16, 17, 18, 19, 23

# NACC 预训练：数据比 ROSMAP 大得多，batch 256（按长度分桶），最多 40 个 epoch、耐心 6
PRE = {"epochs": 40, "patience": 6, "bs": 256}
# seqar（VISTA）：上限放到 100（transformer-sim 的最好 epoch 离 40 太近）；什么时候停仍由耐心 6 决定
PRE_EPOCHS = {"seqar": 100}
SMOKE_EPOCHS = int(os.environ.get("NACC_SMOKE_EPOCHS", "0"))      # 冒烟测试：> 0 时所有训练只跑这么多个 epoch
FT_LR_SCALES = (1.0, 1 / 3)
V_KEYS = ("X", "M", "L", "Y", "delta")            # 第二维是访视的张量（截 batch 时一起截）
LOCK_STALE = 900                                 # 查不到持锁的作业时：锁文件 15 分钟没更新 = 上一个作业已经不在了


def load_nacc():
    Z = np.load(os.path.join(NACC_DIR, "nacc_data.npz"))
    return {s: {k: Z[f"{s}_{k}"] for k in ("X", "M", "S", "L", "Y", "T", "pid")} for s in ("train", "test")}


def nacc_view(D):
    """ROSMAP 的一段 -> NACC 能看到的样子：NACC 没有的输入改成"没测"（手工历史特征同样改成"从没测过"）。只给零样本读出用。"""
    X, L, S = D["X"].copy(), D["L"].copy(), D["S"].copy()
    never = np.array([0, 0, 10, 0, 0, 0, 0, 0], np.float32)              # models_code/input_data.l2c_features 对空历史的输出
    for j in NACC_MISSING_CONT:
        X[..., j] = 0.0
        X[..., I_MC + j] = 1.0
        L[..., j * 8:(j + 1) * 8] = never
    for j in NACC_MISSING_STATE:
        X[..., I_XS + j] = 0.0
        X[..., I_MS + j] = 1.0
    S[:, S_ALC], S[:, S_ALC_NA], S[:, S_MAP], S[:, S_LATC], S[:, S_PREV_CLAUD] = 0.0, 1.0, 0.0, 0.0, 0.0
    return dict(D, X=X, L=L, S=S)


# ============================================================================== 网络（seq、seqsim）：构造、张量、损失


def x_mean_of(D):
    """GRU-D 的 x̄（缺失时往它衰减）：和 sequence.run_sequence 同一个算法（训练段上观测值的平均）。"""
    val = np.concatenate([D["X"][..., :B.N_CONT], D["X"][..., I_XS:I_MS]], -1)
    m = np.concatenate([D["X"][..., I_MC:I_XS] < 0.5, D["X"][..., I_MS:B.I_EVER] < 0.5], -1)
    return ((val * m).sum((0, 1)) / np.maximum(m.sum((0, 1)), 1)).astype(np.float32)


def build(fam, key, x_mean=None):
    """fam = seq / seqsim / seqar（key = 编码器种类；GRU-D 要 x_mean）。"""
    import sequence as SQ
    hp = SQ.SEQ_HP[key]
    if fam == "seq":
        return SQ.MultiTask(SQ.make_encoder(key, hp["d"], hp["dropout"], x_mean), hp["d"])
    import rollout_versions as RV
    if fam == "seqar":                                                 # 风险头只有"下一年内死亡 / AD"两个输出（SeqARModel.cols）
        return RV.build_seqsim(key, None, len(RV.SeqARModel.cols))[0]
    return RV.build_seqsim(key, None)[0]


def hp_of(fam, key):
    """(lr, wd, batch, 最多 epoch, 耐心, 余弦退火)：sequence.run_sequence 的默认。"""
    import sequence as SQ
    hp = SQ.SEQ_HP[key]
    return hp["lr"], hp["wd"], 64, SMOKE_EPOCHS or 150, 15, False


def tensors(fam, D, device):
    import sequence as SQ
    return SQ.tensors(D, device)


def take(T, idx, trim=True):
    """一批人；trim：访视那一维截到这批里最长的记录（填充位置本来就被遮住）。"""
    idx = torch.as_tensor(idx, device=T["X"].device)
    V = int(T["M"][idx].sum(1).max()) if trim else T["X"].shape[1]
    return {k: (v[idx, :V] if k in V_KEYS else v[idx]) for k, v in T.items()}


def loss_of(fam, model, b):
    if fam == "seq":
        return model.loss(b, model(b))
    import rollout_versions as RV
    return RV.seqsim_loss(b, model(b), RV.SeqARModel.cols if fam == "seqar" else None)


@torch.no_grad()
def eval_of(fam, model, T, bs=256):
    """早停准则：每 256 人一批的损失平均（sequence.fit）。"""
    model.eval()
    ls = []
    for s in range(0, len(T["X"]), bs):
        b = take(T, np.arange(s, min(s + bs, len(T["X"]))))
        ls.append(float(loss_of(fam, model, b)))
    return float(np.mean(ls))


@torch.no_grad()
def risk_of(fam, model, T, bs=256):
    """(人, 访视, 6, 10) 的累计风险（截过的 batch 补回原来的访视数）。"""
    model.eval()
    n, V = T["M"].shape
    R = np.zeros((n, V, B.NQ, B.NH), np.float32)
    for s in range(0, n, bs):
        b = take(T, np.arange(s, min(s + bs, n)))
        out = model(b)
        r = model.risk(out) if fam == "seq" else torch.sigmoid(out["risk"]).reshape(*out["risk"].shape[:2], B.NQ, B.NH)
        R[s: s + len(r), : r.shape[1]] = r.cpu().numpy()
    return R


def batches(lengths, bs, rng, bucket):
    n = len(lengths)
    perm = rng.permutation(n)
    if not bucket:
        return [perm[s: s + bs] for s in range(0, n, bs)]
    out = []
    for c in range(0, n, bs * 50):                                 # 50 个 batch 一组，组内按长度排序再切
        ch = perm[c: c + bs * 50]
        ch = ch[np.argsort(lengths[ch], kind="stable")]
        out += [ch[s: s + bs] for s in range(0, len(ch), bs)]
    return [out[k] for k in rng.permutation(len(out))]


def train_loop(fam, model, tr, te, lr, wd, bs, epochs, patience, seed, cosine, bucket=False, partial=None, lock=None):
    """partial：每个 epoch 之后把整个训练状态存下来（续训用）；lock：每个 epoch 摸一下锁文件（证明还活着）。"""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if cosine else None
    ep0, best, best_ep, bad, best_state = 0, np.inf, -1, 0, None
    if partial and os.path.exists(partial):
        st = torch.load(partial, map_location=tr["X"].device)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        if sched is not None:
            sched.load_state_dict(st["sched"])
        ep0, best, best_ep, bad, best_state = st["ep"], st["best"], st["best_ep"], st["bad"], st["best_state"]
        print(f"    resumed at epoch {ep0} (best {best:.5f} @ {best_ep})", flush=True)
    lengths = tr["M"].sum(1).cpu().numpy()
    t0 = time.time()
    ep = ep0 - 1
    for ep in range(ep0, epochs):
        if bad >= patience:
            break
        model.train()
        for idx in batches(lengths, bs, np.random.default_rng([seed, ep]), bucket):
            loss = loss_of(fam, model, take(tr, idx))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if sched is not None:
            sched.step()
        vl = eval_of(fam, model, te)
        if vl < best - 1e-5:
            best, best_ep, bad = vl, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if partial:
            st = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict() if sched else None,
                  "ep": ep + 1, "best": best, "best_ep": best_ep, "bad": bad, "best_state": best_state}
            torch.save(st, partial + ".tmp")
            os.replace(partial + ".tmp", partial)
            print(f"    epoch {ep}: test loss {vl:.5f} (best {best:.5f} @ {best_ep}; {(time.time() - t0) / 60:.1f} min)", flush=True)
        if lock:
            os.utime(lock)
    model.load_state_dict(best_state)
    return {"best_test_loss": best, "best_epoch": best_ep, "epochs_run": ep + 1}


# ============================================================================== NACC 预训练（每个 seed 一个检查点）


def _ckpt(tag, seed):
    os.makedirs(CKPT, exist_ok=True)
    return os.path.join(CKPT, f"{tag}_s{seed}.pt"), os.path.join(CKPT, f"{tag}_s{seed}.json")


def _job_alive(jid):
    """持锁的 Slurm 作业还在不在：True / False；查不了（不在 Slurm 里跑）返回 None。"""
    import subprocess
    if not jid.isdigit():
        return None
    try:
        r = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%T"], capture_output=True, text=True, timeout=60)
    except Exception:
        return None
    if r.returncode != 0:
        return False if "Invalid job id" in r.stderr else None
    return r.stdout.strip() in ("PENDING", "RUNNING", "CONFIGURING", "SUSPENDED", "REQUEUED", "RESIZING")


def _claim(lock):
    """拿到锁 = True。锁里记着持锁的作业号：那个作业已经结束（超时 / 被杀）就接手，接着它存下的进度训。"""
    me = os.environ.get("SLURM_JOB_ID", f"pid{os.getpid()}")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, me.encode())
        os.close(fd)
        return True
    except FileExistsError:
        holder = open(lock).read().strip()
        alive = _job_alive(holder)
        if alive is False or (alive is None and time.time() - os.path.getmtime(lock) > LOCK_STALE):
            with open(lock, "w") as f:
                f.write(me)
            print(f"    taking over lock from {holder}", flush=True)
            return True
        return False


def tag_of(fam, key):
    return f"{fam}-{key}"


def load_pretrained(fam, key, seed, device):
    p, pj = _ckpt(tag_of(fam, key), seed)
    if not os.path.exists(p):
        raise FileNotFoundError(f"没有预训练检查点 {p}：先跑 ris/pretrain.sbatch")
    m = build(fam, key, np.zeros(B.N_CONT + B.N_STATE, np.float32)).to(device)       # x_mean 是缓冲区，从检查点读回来
    m.load_state_dict(torch.load(p, map_location=device))
    return m, json.load(open(pj))


def pretrain(fam, key, seed, device):
    """NACC 上从零训练一个 seed。已经有检查点 -> 直接返回它的记录；别的作业正在训 -> 跳过（返回 None）。"""
    p, pj = _ckpt(tag_of(fam, key), seed)
    if os.path.exists(p):
        return json.load(open(pj))
    lock = p + ".lock"
    if not _claim(lock):
        print(f"  {tag_of(fam, key)} seed {seed}: another job holds the lock, skipping", flush=True)
        return None
    t0 = time.time()
    Dn = load_nacc()
    tr, te = tensors(fam, Dn["train"], device), tensors(fam, Dn["test"], device)
    torch.manual_seed(seed)
    model = build(fam, key, x_mean_of(Dn["train"])).to(device)
    lr, wd, _, _, _, cosine = hp_of(fam, key)
    max_ep = SMOKE_EPOCHS or PRE_EPOCHS.get(fam, PRE["epochs"])
    info = train_loop(fam, model, tr, te, lr, wd, PRE["bs"], max_ep, PRE["patience"], seed, cosine,
                      bucket=True, partial=p + ".partial", lock=lock)
    info.update(minutes_this_job=(time.time() - t0) / 60, n_params=sum(q.numel() for q in model.parameters()),
                n_train_persons=int(len(Dn["train"]["X"])), device=str(device), batch=PRE["bs"], max_epochs=max_ep,
                lr=lr, patience=PRE["patience"])
    torch.save(model.state_dict(), p + ".tmp")
    json.dump(info, open(pj + ".tmp", "w"), indent=1)
    os.replace(pj + ".tmp", pj)
    os.replace(p + ".tmp", p)
    for f in (p + ".partial", lock):
        if os.path.exists(f):
            os.remove(f)
    print(f"  pretrained {tag_of(fam, key)} seed {seed}: NACC test loss {info['best_test_loss']:.4f} @ {info['best_epoch']} "
          f"({info['minutes_this_job']:.1f} min)", flush=True)
    return info


# ============================================================================== ROSMAP 上训练 / 微调（几个 seed 并行）


def _interop1():
    """子进程里只设一次（进程池会把同一个子进程再派给下一个任务，第二次设会报错）。"""
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _threads(n_workers):
    """每个子进程几个线程：按这个作业实际分到的核（cgroup 的 affinity），不是整台机器的核数（os.cpu_count 会超额订阅）。"""
    if torch.cuda.is_available():
        return 2
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 4
    return max(1, n // n_workers)


def member(fam, key, init, seed, Ds, extras, n_workers):
    """一个 seed：scratch / nacc（两个学习率挑一个）/ zeroshot -> test、val、扰动记录上的 (行, 6, 10) 风险。在子进程里跑。"""
    torch.set_num_threads(_threads(n_workers))
    _interop1()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    T = {s: tensors(fam, D, dev) for s, D in Ds.items()}
    Tex = [(tensors(fam, Dp, dev), np.asarray(vL)) for Dp, vL in extras]
    lr, wd, bs, epochs, patience, cosine = hp_of(fam, key)
    xm = x_mean_of(Ds["train"])
    torch.manual_seed(seed)
    if init == "scratch":
        model = build(fam, key, xm).to(dev)
        info = dict(train_loop(fam, model, T["train"], T["test"], lr, wd, bs, epochs, patience, seed, cosine), lr=lr)
    else:
        pre, pinfo = load_pretrained(fam, key, seed, dev)
        if init == "zeroshot":
            model, info = pre, {"test_loss": eval_of(fam, pre, T["test"]), "pretrain": pinfo}
        else:
            state0 = {k: v.detach().clone() for k, v in pre.state_dict().items()}
            best = None
            for sc in FT_LR_SCALES:
                torch.manual_seed(seed)
                m = build(fam, key, xm).to(dev)
                m.load_state_dict(state0)
                if key == "grud":                                       # 微调：缺失往 ROSMAP 的均值衰减（预训练时是 NACC 的）
                    m.enc.x_mean.copy_(torch.as_tensor(xm, device=dev))
                inf = dict(train_loop(fam, m, T["train"], T["test"], lr * sc, wd, bs, epochs, patience, seed, cosine), lr=lr * sc)
                if best is None or inf["best_test_loss"] < best[1]["best_test_loss"]:
                    best = (m, inf)
            model, info = best
            info["pretrain"] = pinfo
    rows = lambda R, D: R[D["M"]]                                     # noqa: E731  站点表的顺序（np.where(M)）
    out = {"test": rows(risk_of(fam, model, T["test"]), Ds["test"]), "val": rows(risk_of(fam, model, T["val"]), Ds["val"]),
           "ex": [risk_of(fam, model, Tp)[np.arange(len(vL)), vL] for Tp, vL in Tex]}
    info.update(seed=seed, minutes=(time.time() - t0) / 60, device=dev)
    return out, info


def run_members(fam, key, kw, Z, log, extras=()):
    """5 个 seed 并行（spawn 子进程；GPU 上共用一块卡），风险取平均 = 集成。"""
    import concurrent.futures as cf
    import multiprocessing as mp
    init, seeds = kw["init"], kw.get("seeds", 5)
    Ds = {s: B.split(Z, s) for s in ("train", "test", "val")}
    if init == "zeroshot":
        Ds = {s: nacc_view(D) for s, D in Ds.items()}
        extras = [(nacc_view(Dp), vL) for Dp, vL in extras]
    extras = [(Dp, np.asarray(vL)) for Dp, vL in extras]
    lm_te, lm_va = B.landmarks(Ds["test"], "cur"), B.landmarks(Ds["val"], "cur")
    res = {}
    with cf.ProcessPoolExecutor(max_workers=seeds, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(member, fam, key, init, s, Ds, extras, seeds): s for s in range(seeds)}
        for f in cf.as_completed(futs):
            out, info = f.result()
            s = futs[f]
            info["test_macro_auc"] = B.macro_auc(lm_te["Y"], out["test"])
            res[s] = out
            log.append(info)
            print(f"  {tag_of(fam, key)} {init} seed {s}: test macro AUC {info['test_macro_auc']:.4f} "
                  f"({info['minutes']:.1f} min, {info['device']})", flush=True)
    R = [res[s] for s in sorted(res)]
    ex_mean = [np.mean([r["ex"][k] for r in R], 0) for k in range(len(extras))]
    return {"val": (lm_va, np.mean([r["val"] for r in R], 0)), "test": (lm_te, np.mean([r["test"] for r in R], 0)),
            "extras": ex_mean}


def run_seq(kw, Z, log, device="cpu", extras=()):
    """kw = {"kind": "transformer", "init": ..., "seeds": 5}。scratch = sequence.run_sequence 同一个配方，多几个 seed。"""
    import sequence as SQ
    r = run_members("seq", kw["kind"], kw, Z, log, extras)
    r["chosen"] = {"kind": kw["kind"], "init": kw["init"], "seeds": kw.get("seeds", 5), **SQ.SEQ_HP[kw["kind"]],
                   "ft_lr_scales": list(FT_LR_SCALES), "pretrain": PRE}
    return r


# ============================================================================== 能生成未来的两个：transformer-sim、ETHOS


class SeqSimNACC:
    """rollout_versions.SeqSimModel（风险头 + 下一次访视分布头）：NACC 预训练的检查点 -> ROSMAP 微调；之后的模拟和原来完全一样。"""

    def __init__(self, kind, device):
        import rollout_versions as RV
        self.inner = RV.SeqSimModel(kind, device)
        self.kind, self.device = kind, device

    def fit(self, Dtr, Dte, seed=0):
        t0 = time.time()
        tr, te = tensors("seqsim", Dtr, self.device), tensors("seqsim", Dte, self.device)
        pre, pinfo = load_pretrained("seqsim", self.kind, seed, self.device)
        state0 = {k: v.detach().clone() for k, v in pre.state_dict().items()}
        lr, wd, bs, epochs, patience, cosine = hp_of("seqsim", self.kind)
        best = None
        for sc in FT_LR_SCALES:
            torch.manual_seed(seed)
            m = build("seqsim", self.kind).to(self.device)
            m.load_state_dict(state0)
            inf = dict(train_loop("seqsim", m, tr, te, lr * sc, wd, bs, epochs, patience, seed, cosine), lr=lr * sc)
            if best is None or inf["best_test_loss"] < best[1]["best_test_loss"]:
                best = (m, inf)
        self.inner.net = best[0].eval()
        self.info = dict(best[1], pretrain=pinfo, minutes=(time.time() - t0) / 60)
        return self

    def sim(self, start, n_mc, rng, norm):
        return self.inner.sim(start, n_mc, rng, norm)


class SeqARNACC:
    """选定的 transformer（VISTA，rollout_versions.SeqARModel）：NACC 预训练的检查点（同一个 seed）-> ROSMAP 微调（2026-11-10 加）。
    微调和从头训练的 VISTA 用同一个训练函数、同一组超参（sequence.fit：batch 64、最多 150 个 epoch、耐心 15、按 test 段损失早停），
    只差两点：起点是 NACC 预训练的权重；学习率在 {原值, 原值 / 3} 里按 test 段损失挑（和表 5 的 -nacc 一样；对照没有这一步，
    所以这一臂多一点挑选的余地，对"没有提升"的结论是保守的）。之后的模拟（sim）和 VISTA 完全一样。"""

    def __init__(self, kind, device):
        import rollout_versions as RV
        self.inner = RV.SeqARModel(kind, device)
        self.kind, self.device = kind, device

    def fit(self, Dtr, Dte, seed=0, max_epochs=150, patience=15):
        import rollout_versions as RV
        import sequence as SQ
        t0 = time.time()
        tr, te = SQ.tensors(Dtr, self.device), SQ.tensors(Dte, self.device)
        pre, pinfo = load_pretrained("seqar", self.kind, seed, self.device)
        state0 = {k: v.detach().clone() for k, v in pre.state_dict().items()}
        hp = SQ.SEQ_HP[self.kind]
        cols = RV.SeqARModel.cols
        tries, best = [], None
        for sc in FT_LR_SCALES:
            torch.manual_seed(seed)
            m = build("seqar", self.kind).to(self.device)
            m.load_state_dict(state0)
            bl, n_ep = SQ.fit(m, tr, te, lambda b, o: RV.seqsim_loss(b, o, cols), hp["lr"] * sc, hp["wd"], 64,
                              max_epochs, patience, seed)
            tries.append({"lr": hp["lr"] * sc, "best_test_loss": bl, "epochs": n_ep})
            if best is None or bl < best[1]["best_test_loss"]:
                best = (m, tries[-1])
        self.inner.net = best[0].eval()
        self.info = dict(best[1], tries=tries, pretrain=pinfo, hp=hp, init="nacc", minutes=(time.time() - t0) / 60)
        return self

    def sim(self, start, n_mc, rng, norm):
        return self.inner.sim(start, n_mc, rng, norm)


def ethos_loop(model, tr, te, device, seed, lr, epochs, patience, bs=64, partial=None, lock=None):
    """ethos.fit 的训练循环（下一个 token 的交叉熵，AdamW wd 0.1，梯度裁剪 1.0，按 test 段损失早停），加上按长度分桶和续训。"""
    import torch.nn.functional as Fn
    import ethos as ETH
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    def loss_of(batch):
        a = torch.as_tensor(ETH._pad_right(batch), device=device)
        km = a > 0
        pos = torch.arange(a.shape[1], device=device)[None].expand_as(a)
        out = model(a[:, :-1], pos[:, :-1], km[:, :-1])
        tgt = a[:, 1:].masked_fill(~km[:, 1:], -100)
        return Fn.cross_entropy(out.reshape(-1, out.shape[-1]), tgt.reshape(-1), ignore_index=-100)

    ep0, best, best_ep, bad, best_state = 0, np.inf, -1, 0, None
    if partial and os.path.exists(partial):
        st = torch.load(partial, map_location=device)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        ep0, best, best_ep, bad, best_state = st["ep"], st["best"], st["best_ep"], st["bad"], st["best_state"]
        print(f"    resumed ethos at epoch {ep0} (best {best:.4f} @ {best_ep})", flush=True)
    lengths = np.array([len(x) for x in tr])
    te_order = np.argsort([len(x) for x in te], kind="stable")
    t0 = time.time()
    ep = ep0 - 1
    for ep in range(ep0, epochs):
        if bad >= patience:
            break
        model.train()
        for idx in batches(lengths, bs, np.random.default_rng([seed, ep]), bucket=True):
            l = loss_of([tr[k] for k in idx])
            opt.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([float(loss_of([te[k] for k in te_order[s: s + 64]])) for s in range(0, len(te), 64)]))
        if vl < best - 1e-4:
            best, best_ep, bad = vl, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if partial:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "ep": ep + 1, "best": best, "best_ep": best_ep,
                        "bad": bad, "best_state": best_state}, partial + ".tmp")
            os.replace(partial + ".tmp", partial)
        print(f"    ethos epoch {ep}: test loss {vl:.4f} (best {best:.4f} @ {best_ep}; {(time.time() - t0) / 60:.1f} min)", flush=True)
        if lock:
            os.utime(lock)
    model.load_state_dict(best_state)
    return {"best_test_loss": best, "best_epoch": best_ep, "epochs_run": ep + 1, "batch": bs}


class EthosNACC:
    """ETHOS：分词（分位数切点、词表）在 ROSMAP train 上定；GPT 先在 NACC 的同一种 token 序列上训练，再在 ROSMAP 上微调。"""

    def __init__(self, meta, device):
        import ethos as ETH
        self.inner = ETH.Ethos(meta, device)
        self.meta, self.device = meta, device

    @staticmethod
    def pretrain(Dtr_rosmap, meta, seed, device):
        """NACC 上训练 GPT（和 ethos.fit 同一个损失、优化器），按序列长度分桶、每个 epoch 存续训检查点、有锁（和 pretrain() 一样）。
        NACC 的 token 序列只算一次，存在 $ZOO_RUN_DIR/nacc/ethos_tokens.pkl（人级，只留在 RIS）。"""
        import pickle
        import ethos as ETH
        import simulator as SIM
        p, pj = _ckpt("ethos", seed)
        if os.path.exists(p):
            return json.load(open(pj))
        lock = p + ".lock"
        if not _claim(lock):
            print(f"  ethos seed {seed}: another job holds the lock, skipping", flush=True)
            return None
        t0 = time.time()
        tok = ETH.Tokenizer(Dtr_rosmap, meta, SIM.norm_info(meta))
        cache = os.path.join(NACC_DIR, "ethos_tokens.pkl")
        if os.path.exists(cache):
            tr, te = pickle.load(open(cache, "rb"))
        else:
            Dn = load_nacc()
            tr, te = ([tok.person(Dn[s], i)[:ETH.MAX_LEN] for i in range(len(Dn[s]["pid"]))] for s in ("train", "test"))
            pickle.dump((tr, te), open(cache + ".tmp", "wb"))
            os.replace(cache + ".tmp", cache)
        torch.manual_seed(seed)
        model = ETH.GPT(len(tok.V)).to(device)
        info = ethos_loop(model, tr, te, device, seed, 1e-3, SMOKE_EPOCHS or PRE["epochs"], 3, partial=p + ".partial", lock=lock)
        info.update(minutes_this_job=(time.time() - t0) / 60, train_tokens=int(sum(len(x) for x in tr)), n_train_persons=len(tr))
        torch.save(model.state_dict(), p + ".tmp")
        json.dump(info, open(pj + ".tmp", "w"), indent=1)
        os.replace(pj + ".tmp", pj)
        os.replace(p + ".tmp", p)
        for f in (p + ".partial", lock):
            if os.path.exists(f):
                os.remove(f)
        return info

    def fit(self, Dtr, Dte, seed=0):
        import ethos as ETH
        t0 = time.time()
        E = self.inner
        E.tok = ETH.Tokenizer(Dtr, self.meta, E.norm)
        pinfo = self.pretrain(Dtr, self.meta, seed, self.device)
        init = torch.load(_ckpt("ethos", seed)[0], map_location=self.device)
        best = None
        for lr in (1e-3, 1e-3 / 3):
            m, inf = ETH.fit(E.tok, Dtr, Dte, self.device, seed=seed, init=init, lr=lr)
            if best is None or inf["best_test_loss"] < best[1]["best_test_loss"]:
                best = (m, dict(inf, lr=lr))
        E.model = best[0]
        self.info = dict(best[1], pretrain=pinfo, minutes=(time.time() - t0) / 60, vocab=len(E.tok.V))
        return self

    def sim(self, start, n_mc, rng, norm):
        return self.inner.sim(start, n_mc, rng, norm)


# ============================================================================== 入口：sbatch ris/pretrain.sbatch


def _pretrain_one(w, seed, n_workers):
    torch.set_num_threads(_threads(n_workers))
    _interop1()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if w == "ethos":
        Z, meta = B.load()
        return EthosNACC.pretrain(B.split(Z, "train"), meta, seed, dev)
    fam, key = {"seq": ("seq", "transformer"), "grud": ("seq", "grud"), "tlstm": ("seq", "tlstm"), "seqsim": ("seqsim", "transformer"),
                "seqar": ("seqar", "transformer")}[w]
    return pretrain(fam, key, seed, dev)


def pretrain_main():
    """把检查点先训好（之后的打包盒作业只读检查点）。几个 (模型, seed) 在同一个作业里并行（spawn 子进程）。"""
    import argparse
    import concurrent.futures as cf
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", nargs="+", default=["seq"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    jobs = [(w, s) for w in args.what for s in (args.seeds if w not in ("seqsim", "ethos") else [0])]
    with cf.ProcessPoolExecutor(max_workers=len(jobs), mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_pretrain_one, w, s, len(jobs)): (w, s) for w, s in jobs}
        for f in cf.as_completed(futs):
            w, s = futs[f]
            info = f.result()
            short = {k: v for k, v in (info or {}).items() if not isinstance(v, (list, dict))}
            print(f"{w} seed {s}: {json.dumps(short)}", flush=True)
    print("pretraining done", flush=True)


if __name__ == "__main__":
    pretrain_main()
