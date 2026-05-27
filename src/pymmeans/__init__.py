"""pymmeans — estimated marginal means for Python."""

from pymmeans.adapters import (
    LinearmodelsAdapter,
    ModelAdapter,
    StatsmodelsAdapter,
    register_adapter,
)
from pymmeans.adjustments import adjust_pvalues
from pymmeans.cld import cld
from pymmeans.contrasts import (
    ContrastResult,
    EmmList,
    contrast,
    effect_size,
    pairs,
    rbind,
)
from pymmeans.diagnostics import Check, HealthReport, health_check
from pymmeans.emmeans import EMMResult, emmeans
from pymmeans.joint import eta_squared, joint_tests
from pymmeans.ml import (
    MLEMMResult,
    MLPredictInfo,
    from_predict,
    ml_contrast,
    ml_emmeans,
    ml_pairs,
)
from pymmeans.multinom import multinom_emmeans
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
from pymmeans.plotting import emmip, plot, pwpp
from pymmeans.posterior import (
    PosteriorInfo,
    from_pymc,
    posterior_emm_summary,
    posterior_emmeans,
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
    NonLogContrastBiasAdjustError,
    Transform,
    detect_transform,
    make_tran,
    regrid,
    regrid_response,
)
from pymmeans.trends import emtrends
from pymmeans.utils import from_fitted, from_linearmodels, from_statsmodels

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

__version__ = "0.1.6"

__all__ = [
    "BoundaryFitError",
    "Check",
    "ContrastResult",
    "EMMResult",
    "EmmList",
    "FtestResult",
    "HealthReport",
    "KRDiagnostics",
    "LinearmodelsAdapter",
    "MLEMMResult",
    "MLPredictInfo",
    "ModelAdapter",
    "NonLogContrastBiasAdjustError",
    "PBmodcompResult",
    "PosteriorInfo",
    "RefGrid",
    "StatsmodelsAdapter",
    "SurveyDesign",
    "Transform",
    "__version__",
    "adjust_pvalues",
    "apply_kenward_roger",
    "apply_satterthwaite",
    "as_r_frame",
    "bootstrap_ci",
    "cld",
    "confint",
    "contrast",
    "ddf_lb",
    "design_corrected_vcov",
    "detect_transform",
    "effect_size",
    "emm_options",
    "emmeans",
    "emmip",
    "emmobj",
    "emtrends",
    "eta_squared",
    "from_fitted",
    "from_linearmodels",
    "from_predict",
    "from_pymc",
    "from_statsmodels",
    "from_survey",
    "get_emm_option",
    "get_kr",
    "get_lsm_option",
    "health_check",
    "joint_tests",
    "kenward_roger_vcov",
    "krmodcomp",
    "lsm",
    "lsm_options",
    "lsmeans",
    "lsmip",
    "lstrends",
    "make_tran",
    "ml_contrast",
    "ml_emmeans",
    "ml_pairs",
    "multinom_emmeans",
    "ordinal_emmeans",
    "pairs",
    "pbmodcomp",
    "permutation_test",
    "plot",
    "posterior_emm_summary",
    "posterior_emmeans",
    "pwpm",
    "pwpp",
    "qdrg",
    "rbind",
    "ref_grid",
    "register_adapter",
    "regrid",
    "regrid_response",
    "reset_emm_options",
    "satmodcomp",
    "satterthwaite_df",
    "set_emm_options",
    "summary",
    "test",
    "update",
]
