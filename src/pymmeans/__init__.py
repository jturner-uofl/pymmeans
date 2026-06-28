"""pymmeans — estimated marginal means for Python."""

from pymmeans.adapters import (
    LinearmodelsAdapter,
    ModelAdapter,
    PyFixestAdapter,
    StatsmodelsAdapter,
    register_adapter,
)
from pymmeans.adjustments import adjust_pvalues
from pymmeans.aft import from_aft
from pymmeans.cld import cld
from pymmeans.comparisons import (
    ComparisonsResult,
    avg_comparisons,
    comparisons,
)
from pymmeans.conformal import (
    ConformalCounterfactualResult,
    ConformalPIResult,
    conformal_counterfactual_pi,
    split_conformal_pi,
)
from pymmeans.contrasts import (
    CONTRAST_METHODS,
    ContrastResult,
    EmmList,
    contrast,
    effect_size,
    opoly,
    pairs,
    rbind,
    register_contrast_method,
)
from pymmeans.datagrid import datagrid
from pymmeans.diagnostics import Check, HealthReport, health_check
from pymmeans.double_ml import AIPWResult, aipw_ate, cross_fit_ml_emmeans
from pymmeans.emmeans import EMMResult, emmeans
from pymmeans.grid_ops import (
    add_grouping,
    comb_facs,
    force_regular,
    permute_levels,
    split_fac,
)
from pymmeans.hypotheses import HypothesisResult, hypotheses
from pymmeans.imputation import PooledImputationResult, pool_imputed
from pymmeans.joint import eta_squared, joint_tests
from pymmeans.ml import (
    MLEMMResult,
    MLMarginalResult,
    MLPredictInfo,
    from_predict,
    ml_avg_comparisons,
    ml_avg_slopes,
    ml_contrast,
    ml_emmeans,
    ml_pairs,
)
from pymmeans.multinom import multinom_emmeans
from pymmeans.multivariate import (
    MultivariateEMM,
    MultivariateInfo,
    from_multivariate,
    multivariate_emmeans,
    mvcontrast,
)
from pymmeans.options import (
    emm_options,
    get_emm_option,
    reset_emm_options,
    set_emm_options,
)
from pymmeans.ordinal import ordinal_emmeans
from pymmeans.pbktest import (
    FtestResult,
    KRDiagnostics,
    ddf_lb,
    get_kr,
    krmodcomp,
    satmodcomp,
)
from pymmeans.pbmodcomp import PBmodcompResult, pbmodcomp
from pymmeans.plotting import (
    emmip,
    plot,
    plot_comparisons,
    plot_predictions,
    plot_slopes,
    pwpp,
)
from pymmeans.posterior import (
    PosteriorInfo,
    from_arviz,
    from_pymc,
    posterior_emm_summary,
    posterior_emmeans,
)
from pymmeans.predictions import (
    PredictionsResult,
    avg_predictions,
    predictions,
)
from pymmeans.pwpm import pwpm
from pymmeans.qdrg import emmobj, qdrg
from pymmeans.ref_grid import RefGrid, ref_grid
from pymmeans.satterthwaite import (
    BoundaryFitError,
    apply_kenward_roger,
    apply_satterthwaite,
    kenward_roger_vcov,
    satterthwaite_df,
)
from pymmeans.sensitivity import EValueResult, e_value
from pymmeans.slopes import SlopesResult, avg_slopes, slopes
from pymmeans.summary import bootstrap_ci, permutation_test
from pymmeans.summary_layer import (
    as_r_frame,
    confint,
    summary,
    test,
    update,
)
from pymmeans.survey import SurveyDesign, design_corrected_vcov, from_survey
from pymmeans.transforms import (
    TRANSFORMS,
    NonLogContrastBiasAdjustError,
    Transform,
    detect_transform,
    make_tran,
    register_transform,
    regrid,
    regrid_response,
)
from pymmeans.trends import emtrends
from pymmeans.utils import (
    from_fitted,
    from_glmgam,
    from_linearmodels,
    from_pyfixest,
    from_statsmodels,
)

# R `lsmeans` package backward-compat aliases.
#
# `pymmeans` exposes the four most-used analysis-facing aliases
# (`lsmeans` / `lsm` / `lstrends` / `lsmip`) plus the `lsm_options`
# / `get_lsm_option` pair, all of which are documented in R
# `lsmeans` v2.30-2 (March 2025) as transitional aliases for
# `emmeans` / `emtrends` / `emmip` / `emm_options`. The CRAN
# `transition` help topic states "lsmeans now relies primarily on
# code in the emmeans package".
#
# `pymmeans` implements every function the `lsmeans` R package
# re-exports under its own namespace, with the standard PEP 8
# dot-to-underscore translation for names like `ref.grid` →
# `ref_grid` and `lsm.basis` → `emm_basis`. The low-level
# constructor `lsmobj` is the same entry point as R `emmobj`;
# `pymmeans` ships `emmobj` and treats `lsmobj` as a transitional
# alias. See `docs/r_parity_matrix.md` for the per-function
# coverage table.
lsmeans = emmeans
lsm = emmeans
lstrends = emtrends
lsmip = emmip
lsm_options = emm_options
get_lsm_option = get_emm_option

__version__ = "0.15.1"

__all__ = [
    "CONTRAST_METHODS",
    "TRANSFORMS",
    "AIPWResult",
    "BoundaryFitError",
    "Check",
    "ComparisonsResult",
    "ConformalCounterfactualResult",
    "ConformalPIResult",
    "ContrastResult",
    "EMMResult",
    "EValueResult",
    "EmmList",
    "FtestResult",
    "HealthReport",
    "HypothesisResult",
    "KRDiagnostics",
    "LinearmodelsAdapter",
    "MLEMMResult",
    "MLMarginalResult",
    "MLPredictInfo",
    "ModelAdapter",
    "MultivariateEMM",
    "MultivariateInfo",
    "NonLogContrastBiasAdjustError",
    "PBmodcompResult",
    "PooledImputationResult",
    "PosteriorInfo",
    "PredictionsResult",
    "PyFixestAdapter",
    "RefGrid",
    "SlopesResult",
    "StatsmodelsAdapter",
    "SurveyDesign",
    "Transform",
    "__version__",
    "add_grouping",
    "adjust_pvalues",
    "aipw_ate",
    "apply_kenward_roger",
    "apply_satterthwaite",
    "as_r_frame",
    "avg_comparisons",
    "avg_predictions",
    "avg_slopes",
    "bootstrap_ci",
    "cld",
    "comb_facs",
    "comparisons",
    "confint",
    "conformal_counterfactual_pi",
    "contrast",
    "cross_fit_ml_emmeans",
    "datagrid",
    "ddf_lb",
    "design_corrected_vcov",
    "detect_transform",
    "e_value",
    "effect_size",
    "emm_options",
    "emmeans",
    "emmip",
    "emmobj",
    "emtrends",
    "eta_squared",
    "force_regular",
    "from_aft",
    "from_arviz",
    "from_fitted",
    "from_glmgam",
    "from_linearmodels",
    "from_multivariate",
    "from_predict",
    "from_pyfixest",
    "from_pymc",
    "from_statsmodels",
    "from_survey",
    "get_emm_option",
    "get_kr",
    "get_lsm_option",
    "health_check",
    "hypotheses",
    "joint_tests",
    "kenward_roger_vcov",
    "krmodcomp",
    "lsm",
    "lsm_options",
    "lsmeans",
    "lsmip",
    "lstrends",
    "make_tran",
    "ml_avg_comparisons",
    "ml_avg_slopes",
    "ml_contrast",
    "ml_emmeans",
    "ml_pairs",
    "multinom_emmeans",
    "multivariate_emmeans",
    "mvcontrast",
    "opoly",
    "ordinal_emmeans",
    "pairs",
    "pbmodcomp",
    "permutation_test",
    "permute_levels",
    "plot",
    "plot_comparisons",
    "plot_predictions",
    "plot_slopes",
    "pool_imputed",
    "posterior_emm_summary",
    "posterior_emmeans",
    "predictions",
    "pwpm",
    "pwpp",
    "qdrg",
    "rbind",
    "ref_grid",
    "register_adapter",
    "register_contrast_method",
    "register_transform",
    "regrid",
    "regrid_response",
    "reset_emm_options",
    "satmodcomp",
    "satterthwaite_df",
    "set_emm_options",
    "slopes",
    "split_conformal_pi",
    "split_fac",
    "summary",
    "test",
    "update",
]
