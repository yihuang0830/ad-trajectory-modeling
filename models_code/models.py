"""models.py -- 模型注册表（Delphi 以外的全部）：名字 -> (家族, 实现, 参数, 做法说明, 怎么预测未来)。

名字的写法：<方法>-<输入>。输入 demo = 年龄 + 人口学；cur = 当前这次访视 + 静态背景（不看更早的访视）；
l2c = cur + 手工历史特征（TADPOLE Frog / L2C-FNN 的 8 个/变量）；没有后缀的序列模型看整条访视历史（每次访视 54 维 + 静态背景）。
怎么预测未来（future）：
  rollout   生成模型：从站点起逐年模拟，风险 = 模拟轨迹里事件的比例；也能给 10 年轨迹（fig4、fig5）
  exact     多状态 Markov：转移矩阵连乘精确算出每年的状态分布（也能抽样出轨迹）
  survival  给整条生存曲线 S(t | 站点)，任意 H 年的风险 = 1 − S(H)；不能生成轨迹
  direct    每道题 × 每个时间窗直接输出概率（60 个输出）；不能生成轨迹
"""

AR = ("autoregressive: trained only on the next step (death / AD in the coming year + the next visit's distribution), "
      "every answer by rollout")

# name, family, impl, args, description, future
BASELINES = [
    # --- 站点表上的分类器（每道题 × 每个时间窗一个二分类器）
    ("lr-demo", "landmark classifier", "clf", {"kind": "lr", "feat": "demo"},
     "Logistic regression on age + demographics (sex, education, race, Hispanic, APOE)", "direct"),
    ("lr-cur", "landmark classifier", "clf", {"kind": "lr", "feat": "cur"},
     "Logistic regression on the current visit (landmark logistic, LOCF; van Houwelingen 2007)", "direct"),
    ("lr-l2c", "landmark classifier", "clf", {"kind": "lr", "feat": "l2c"},
     "Logistic regression on current visit + L2C history features", "direct"),
    ("rf-l2c", "landmark classifier", "clf", {"kind": "rf", "feat": "l2c"},
     "Random forest (Breiman 2001) on L2C features", "direct"),
    ("xgb-l2c", "landmark classifier", "clf", {"kind": "xgb", "feat": "l2c"},
     "L2C-XGBoost: TADPOLE winner 'Frog' (Marinescu et al., MELBA 2021; Zhang et al., HBM 2025)", "direct"),
    ("lgbm-l2c", "landmark classifier", "clf", {"kind": "lgbm", "feat": "l2c"},
     "LightGBM (Ke et al., NeurIPS 2017) on L2C features", "direct"),
    ("tabpfn-l2c", "landmark classifier", "clf", {"kind": "tabpfn", "feat": "l2c"},
     "TabPFN v2 tabular foundation model, defaults, local GPU inference (Hollmann et al., Nature 2025)", "direct"),
    ("fnn-l2c", "landmark classifier", "seq", {"kind": "fnn"},
     "L2C-FNN (Zhang et al., Human Brain Mapping 2025): feed-forward net on L2C features", "direct"),
    # --- 固定权重的临床风险评分（不训练；只在 train 上把分数换算成概率，排序 = 原评分）
    ("bdsi", "risk score", "score", {"kind": "bdsi"},
     "Brief Dementia Screening Indicator (Barnes et al., Alzheimers Dement 2014); IADL item unavailable", "direct"),
    ("anu-adri", "risk score", "score", {"kind": "anuadri"},
     "ANU-ADRI, common-variable set + depression (Anstey et al., PLoS ONE 2014; MAP AUC 0.689)", "direct"),
    ("caide", "risk score", "score", {"kind": "caide"},
     "CAIDE dementia risk score without physical activity (Kivipelto et al., Lancet Neurol 2006)", "direct"),
    # --- 站点表上的生存模型（每道题一个，整条生存曲线）
    ("cox-cogn", "landmark survival", "surv", {"kind": "cox", "feat": "cogn"},
     "Landmark Cox on the last global cognition score only (Yang et al., J Alzheimers Dis 2022, ROS/MAP)", "survival"),
    ("cox-super", "landmark survival", "surv", {"kind": "cox", "feat": "super"},
     "Landmark Cox supermodel: current visit x (s, s^2) interactions (van Houwelingen 2007; dynamicLM)", "survival"),
    ("cox-cur", "landmark survival", "surv", {"kind": "cox", "feat": "cur"},
     "Landmark Cox PH on the current visit (LOCF; van Houwelingen 2007; Signorelli & Retif 2025's reference)", "survival"),
    ("cox-l2c", "landmark survival", "surv", {"kind": "cox", "feat": "l2c"},
     "Landmark Cox PH on current visit + L2C history features", "survival"),
    ("rsf-l2c", "landmark survival", "surv", {"kind": "rsf", "feat": "l2c"},
     "Landmark random survival forest (Ishwaran 2008; best model in de Swart et al., Front Neurol 2025)", "survival"),
    ("xgbcox-l2c", "landmark survival", "surv", {"kind": "xgbcox", "feat": "l2c"},
     "Gradient-boosted Cox (XGBoost survival:cox + Breslow baseline)", "survival"),
    ("survivalboost-l2c", "landmark survival", "surv", {"kind": "survivalboost", "feat": "l2c"},
     "SurvivalBoost competing-risk gradient boosting, death competing (Alberge et al., AISTATS 2025; hazardous)",
     "survival"),
    ("deepsurv-l2c", "landmark survival", "surv", {"kind": "deepsurv", "feat": "l2c"},
     "DeepSurv (Katzman et al., BMC Med Res Methodol 2018)", "survival"),
    ("deephit-l2c", "landmark survival", "surv", {"kind": "deephit", "feat": "l2c"},
     "DeepHit (Lee et al., AAAI 2018)", "survival"),
    ("logistichazard-l2c", "landmark survival", "surv", {"kind": "logistichazard", "feat": "l2c"},
     "Logistic-Hazard / Nnet-survival (Gensheimer & Narasimhan 2019; Kvamme & Borgan 2021)", "survival"),
    ("pmf-l2c", "landmark survival", "surv", {"kind": "pmf", "feat": "l2c"},
     "PMF discrete-time survival net (Kvamme & Borgan 2021)", "survival"),
    ("mtlr-l2c", "landmark survival", "surv", {"kind": "mtlr", "feat": "l2c"},
     "N-MTLR (Fotso 2018; Yu et al. 2011)", "survival"),
    # --- 看整条历史的序列模型：自回归，只学下一步（下一年内死亡 / AD + 下一次访视的分布），所有答案从 rollout 来
    #     （rollout_versions.SeqARModel / DDHARModel；2026-10-28 起，原来逐题逐年直接给的版本和它们的 -sim 版都换成这个）
    ("gru", "sequence", "seqar", {"kind": "gru"}, f"GRU (Cho et al. 2014) over all visits, {AR}", "rollout"),
    ("lstm", "sequence", "seqar", {"kind": "lstm"}, f"LSTM (Hochreiter & Schmidhuber 1997) over all visits, {AR}", "rollout"),
    ("transformer", "sequence", "seqar", {"kind": "transformer"},
     f"Causal transformer over visits (visit vector + visit-index embedding), {AR}", "rollout"),
    # 2026-11-05 TRFX 实验的组合（models_code/trfx/）：31 类 MMSE 头 + 不要访视编号嵌入 + transformer -> GRU + 模型补缺失
    ("transformer-gru-fill", "sequence", "trfx", {"config": "s3-b-gru-fill.json"},
     "Causal transformer over visits, TRFX variant s3-b-gru-fill (31-way categorical next-MMSE head, no visit-index "
     "embedding, width 32, transformer -> GRU hybrid encoder, MinimalRNN-style model filling of missing inputs), "
     f"{AR}", "rollout"),
    ("grud", "sequence", "seqar", {"kind": "grud"}, f"GRU-D (Che et al., Scientific Reports 2018), {AR}", "rollout"),
    ("tlstm", "sequence", "seqar", {"kind": "tlstm"}, f"T-LSTM (Baytas et al., KDD 2017), {AR}", "rollout"),
    ("retain", "sequence", "seqar", {"kind": "retain"}, f"RETAIN (Choi et al., NeurIPS 2016), {AR}", "rollout"),
    ("dynamic-deephit", "sequence", "ddhar", {},
     f"Dynamic-DeepHit (Lee, Yoon & van der Schaar, IEEE TBME 2020) with death and AD as the two causes, {AR}", "rollout"),
    # --- 能生成未来的模型
    ("markov", "multistate", "msm", {"cov": False},
     "Discrete-time multistate Markov chain on (AD, MMSE stage, cognition stage) + death; empirical transitions", "exact"),
    ("msm-cov", "multistate", "msm", {"cov": True},
     "Multistate Markov model with transition-specific covariates (age, sex, education, APOE; msm, Jackson 2011)",
     "exact"),
    ("hmm", "hidden Markov", "hmm", {},
     "Latent (hidden) Markov model: 3 forward-only latent cognitive stages + death, Gaussian MMSE / cognition and "
     "new-AD emissions, covariate transitions (Yu et al., Neurobiol Aging 2015, ROS/MAP), EM", "rollout"),
    ("lmm-joint", "joint model", "lmm", {},
     "Two-stage joint model: linear mixed models for MMSE and cognition + discrete-time hazards on current value "
     "and slope (Rizopoulos 2012; Wilson et al. ROS/MAP mixed models)", "rollout"),
    ("ar-lr", "microsimulation", "ar", {"kind": "lr"},
     "Parametric g-formula, natural course (Robins 1986; gfoRmula, McGrath et al., Patterns 2020): linear / logistic "
     "model per time-varying covariate + pooled-logistic AD and death hazards, simulated year by year",
     "rollout"),
    ("ar-lgbm", "microsimulation", "ar", {"kind": "lgbm"},
     "Parametric g-formula with LightGBM covariate and hazard models (same simulator as ar-lr)", "rollout"),
    ("ethos", "generative transformer", "ethos", {},
     "ETHOS (Renc et al., npj Digit Med 2024): GPT over event tokens with time-interval and quantile tokens, "
     "trained from scratch; risk = share of generated futures", "rollout"),
    ("minimalrnn", "generative RNN", "minimalrnn", {},
     "MinimalRNN with model filling (Nguyen et al., NeuroImage 2020; TADPOLE 2nd) + added death head", "rollout"),
    # --- 2026-10-22 加：统计 / 树模型 / 高斯过程里原来缺的几个
    ("fg-cur", "landmark survival", "surv", {"kind": "fg", "feat": "cur"},
     "Landmark Fine-Gray subdistribution hazards, death as the competing event (Fine & Gray, JASA 1999); death itself = Cox",
     "survival"),
    ("fg-l2c", "landmark survival", "surv", {"kind": "fg", "feat": "l2c"},
     "Landmark Fine-Gray on current visit + L2C history features, death competing", "survival"),
    ("xgbaft-l2c", "landmark survival", "surv", {"kind": "xgbaft", "feat": "l2c"},
     "Gradient-boosted accelerated failure time model, interval-censored (XGBoost survival:aft; Barnwal, Cho & Hocking, "
     "JCGS 2022)", "survival"),
    ("lgbmcox-l2c", "landmark survival", "surv", {"kind": "lgbmcox", "feat": "l2c"},
     "Gradient-boosted Cox with LightGBM: custom partial-likelihood objective (LightGBM has no built-in survival "
     "objective; same gradient / Hessian as XGBoost survival:cox) + Breslow baseline", "survival"),
    ("msm-ct", "multistate", "ctmsm", {},
     "Continuous-time multistate Markov model on (AD, MMSE stage, cognition stage) + death: panel-data likelihood with "
     "censored states, transition-type covariate effects, time-varying age (Kalbfleisch & Lawless 1985; msm, Jackson 2011)",
     "exact"),
    ("jm", "joint model", "jm", {},
     "Shared-random-effects joint model, estimated jointly: bivariate linear mixed model (MMSE, cognition; correlated "
     "random intercepts and slopes) + discrete-time AD and death hazards on current value and slope, likelihood "
     "integrated over the random effects (Rizopoulos 2012; JM / JMbayes2)", "rollout"),
    ("gpboost-joint", "joint model", "mixed", {"kind": "gpboost"},
     "GPBoost trajectories: tree-boosted fixed effects + per-person random intercept and slope (Sigrist, JMLR 2022), "
     "then discrete-time AD and death hazards on current value and slope", "rollout"),
    ("merf-joint", "joint model", "mixed", {"kind": "merf"},
     "MERF trajectories: mixed-effects random forest with per-person random intercept and slope (Hajjem, Bellavance & "
     "Larocque 2014), then discrete-time AD and death hazards on current value and slope", "rollout"),
    ("gp-joint", "joint model", "gp", {},
     "Hierarchical Gaussian-process trajectories: population mean + per-person GP (random intercept, slope and "
     "squared-exponential kernels, marginal likelihood; Rasmussen & Williams 2006; personalized GP for AD, Peterson et "
     "al. 2017), then discrete-time AD and death hazards on current value and slope", "rollout"),
]

# --- 不能 rollout 的基线各自的 rollout 版（rollout_versions.py；2026-10-21 加）。lr-l2c、lgbm-l2c 的 rollout 版就是上面的 ar-lr、ar-lgbm。
SIM_OF = {"lr-l2c": "ar-lr", "lgbm-l2c": "ar-lgbm"}


def _sim(orig, impl, kw, desc):
    SIM_OF[orig] = f"{orig}-sim"
    return (f"{orig}-sim", "rollout version", impl, {"orig": orig, **kw}, f"Rollout version of {orig}: {desc}", "rollout")


BASELINES += [
    _sim("lr-demo", "arv", {"cov": "lr", "feat": "demo", "haz": "clf"}, "logistic / linear one-step models on demographics only"),
    _sim("lr-cur", "arv", {"cov": "lr", "feat": "cur", "haz": "clf"}, "logistic / linear one-step models on the current visit"),
    _sim("rf-l2c", "arv", {"cov": "rf", "feat": "l2c", "haz": "clf"}, "random-forest one-step models"),
    _sim("xgb-l2c", "arv", {"cov": "xgb", "feat": "l2c", "haz": "clf"}, "XGBoost one-step models"),
    _sim("tabpfn-l2c", "arv", {"cov": "tabpfn", "feat": "l2c", "haz": "clf", "cov_hp": {"n_estimators": 1, "ctx": 5000}},
         "TabPFN one-step models (covariates: 1 estimator, 5k-row context)"),
    _sim("bdsi", "arv", {"cov": "lr", "feat": "cur", "haz": "score:bdsi"}, "BDSI score as the yearly hazard + linear covariate models"),
    _sim("anu-adri", "arv", {"cov": "lr", "feat": "cur", "haz": "score:anuadri"}, "ANU-ADRI score as the yearly hazard + linear covariate models"),
    _sim("caide", "arv", {"cov": "lr", "feat": "cur", "haz": "score:caide"}, "CAIDE score as the yearly hazard + linear covariate models"),
    _sim("cox-cur", "arv", {"cov": "lr", "feat": "cur", "haz": "surv:cox"}, "Cox 1-year hazards + linear covariate models"),
    _sim("cox-l2c", "arv", {"cov": "lr", "feat": "l2c", "haz": "surv:cox"}, "Cox 1-year hazards + linear covariate models"),
    _sim("cox-cogn", "arv", {"cov": "lr", "feat": "cogn", "haz": "surv:cox"}, "Cox 1-year hazards + linear covariate models, cognition only"),
    _sim("cox-super", "arv", {"cov": "lr", "feat": "super", "haz": "surv:cox"}, "Cox supermodel 1-year hazards + linear covariate models"),
    _sim("rsf-l2c", "arv", {"cov": "rf", "feat": "l2c", "haz": "surv:rsf", "cov_hp": {"leaf": 16}},
         "random survival forest 1-year hazards + random-forest covariate models"),
    _sim("xgbcox-l2c", "arv", {"cov": "xgb", "feat": "l2c", "haz": "surv:xgbcox"}, "XGBoost-Cox 1-year hazards + XGBoost covariate models"),
    _sim("survivalboost-l2c", "arv", {"cov": "hgb", "feat": "l2c", "haz": "surv:survivalboost"},
         "SurvivalBoost 1-year hazards + histogram gradient boosting covariate models"),
] + [_sim(f"{k}-l2c", "arv", {"cov": "mlp", "feat": "l2c", "haz": f"surv:{k}"}, f"{k} 1-year hazards + pycox-default MLP covariate model")
     for k in ("deepsurv", "deephit", "logistichazard", "pmf", "mtlr")] + [
    _sim("fnn-l2c", "seqsim", {"kind": "fnn"}, "same network + next-visit distribution head"),
    _sim("fg-cur", "arv", {"cov": "lr", "feat": "cur", "haz": "surv:fg"}, "Fine-Gray 1-year AD risk + Cox 1-year death + linear covariate models"),
    _sim("fg-l2c", "arv", {"cov": "lr", "feat": "l2c", "haz": "surv:fg"}, "Fine-Gray 1-year AD risk + Cox 1-year death + linear covariate models"),
    _sim("xgbaft-l2c", "arv", {"cov": "xgb", "feat": "l2c", "haz": "surv:xgbaft"}, "XGBoost-AFT 1-year hazards + XGBoost covariate models"),
    _sim("lgbmcox-l2c", "arv", {"cov": "lgbm", "feat": "l2c", "haz": "surv:lgbmcox"}, "LightGBM-Cox 1-year hazards + LightGBM covariate models"),
]

# --- NACC 预训练 -> ROSMAP 微调（pretrain.py；2026-10-22 加）。每个结构三种起点：只在 ROSMAP 上训练（对照）/ NACC 预训练后微调 / 不微调。
NACC_FAMILY = "NACC pretraining"
INIT_DESC = {"scratch": "trained on ROSMAP only (control)", "nacc": "pretrained on NACC, fine-tuned on ROSMAP",
             "zeroshot": "pretrained on NACC, no fine-tuning (inputs NACC lacks set to missing)"}
NACC_RUNS = []
SEQ_DESC = {"transformer": "Causal transformer over visits (same as transformer)", "grud": "GRU-D (same as grud)",
            "tlstm": "T-LSTM, time-aware LSTM (same as tlstm)"}
for _k in SEQ_DESC:
    for _init, _sfx in (("scratch", "-x5"), ("nacc", "-nacc"), ("zeroshot", "-nacc-zs")):
        NACC_RUNS.append((f"{_k}{_sfx}", NACC_FAMILY, "seqn", {"kind": _k, "init": _init, "seeds": 5},
                          f"{SEQ_DESC[_k]}, {INIT_DESC[_init]}; 5-seed ensemble", "direct"))
NACC_RUNS += [
    # transformer-sim（60 个题目头 + 下一次访视头，2026-10-28 以前的做法）只留作 transformer-sim-nacc 的对照
    ("transformer-sim", NACC_FAMILY, "seqsim", {"kind": "transformer", "init": "scratch"},
     "Earlier rollout version of transformer: same network + next-visit distribution head, trained together with the "
     "60 direct-answer outputs (control for transformer-sim-nacc)", "rollout"),
    ("transformer-sim-nacc", NACC_FAMILY, "seqsimnacc", {"kind": "transformer"},
     "transformer-sim (risk head + next-visit head, rollout), pretrained on NACC, fine-tuned on ROSMAP", "rollout"),
    ("ethos-nacc", NACC_FAMILY, "ethosnacc", {},
     "ETHOS (same tokenizer as ethos, fixed on ROSMAP train), GPT pretrained on NACC token sequences, fine-tuned on ROSMAP",
     "rollout"),
]
# 每个 NACC 变体拿来比的对照（同一结构、同一份代码，只差初始化）
NACC_CONTROL = {**{f"{k}{s}": f"{k}-x5" for k in ("transformer", "grud", "tlstm") for s in ("-nacc", "-nacc-zs")},
                "transformer-sim-nacc": "transformer-sim", "ethos-nacc": "ethos"}
BASELINES += NACC_RUNS
NACC_NAMES = {b[0] for b in NACC_RUNS}

# --- 只在 RIS 上量化用的变体：不进 MODELS.md 的表和图（2026-10-29 加）
PROBES = [
    ("minimalrnn-tuned", "generative RNN", "minimalrnn", {"hp": "tuned"},
     "MinimalRNN as minimalrnn, hyperparameters tuned over the paper's search space (Nguyen et al. 2020, Table 2; "
     "random search on the test split instead of HORD, tune_minimalrnn.py)", "rollout"),
]

BY_NAME = {b[0]: b for b in BASELINES + PROBES}
GENERATIVE = [b[0] for b in BASELINES + PROBES if b[5] in ("rollout", "exact")]
GPU = ({b[0] for b in BASELINES + PROBES if b[2] in ("seq", "minimalrnn", "ethos", "seqsim", "seqar", "ddhar", "seqn", "seqsimnacc",
                                             "ethosnacc", "jm", "gp", "ctmsm")}
       | {"tabpfn-l2c", "tabpfn-l2c-sim"})
