"""Read and check the ``equity_report`` section of the analysis settings."""

__all__ = ['equity_report_settings', 'DEFAULT_RETURN_PERIODS']

from ..utils.exceptions import OasisException

SETTINGS_KEY = 'equity_report'
RUNTYPES = ('gul', 'il', 'ri', 'rl')
DEFAULT_RETURN_PERIODS = [10, 25, 50, 100, 200, 250, 500, 1000]
PREMIUM_SCHEMES = ('risk_based', 'flat', 'capped', 'means_tested')

DEFAULTS = {
    'runtype': 'gul',
    'summary_set': 1,
    'group_by': None,
    'income_column': None,
    'burden_threshold': 0.01,
    'income_burden_threshold': 0.05,
    'min_group_size': 10,
    'sample_type': None,
    'equity_weighting': None,
    'premiums': None,
    'data_gaps': None,
    'loss_sharing': None,
    'public_assistance': None,
    'basis_risk': None,
}
LOSS_SHARING_PERSPECTIVES = ('gul', 'il', 'ri')
BASIS_RISK_DEFAULTS = {'loss_threshold': 0.0, 'payout_threshold': 0.0}
ASSISTANCE_DEFAULTS = {'deductible': 0.0, 'share': 1.0, 'limit': None, 'income_threshold': None,
                       'return_periods': DEFAULT_RETURN_PERIODS}
DATA_GAP_DEFAULTS = {'default_vulnerability_ids': [], 'geocode_quality_threshold': None}
WEIGHTING_DEFAULTS = {'eta': 1.5, 'reference_income': 'mean', 'return_periods': DEFAULT_RETURN_PERIODS}
PREMIUM_DEFAULTS = {'loading': 0.0, 'schemes': None}
SCHEME_DEFAULTS = {
    'risk_based': {},
    'flat': {'basis': 'row'},
    'capped': {'cap': None, 'funding': 'subsidy'},
    'means_tested': {'income_threshold': None, 'discount': None, 'funding': 'subsidy'},
}
FUNDING = ('subsidy', 'surcharge')


def _check_keys(section, given, allowed):
    unknown = set(given) - set(allowed)
    if unknown:
        raise OasisException(f"{section} has unknown keys {sorted(unknown)}; expected some of {sorted(allowed)}")


def _positive(section, name, value, allow_zero=False):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0 or (value == 0 and not allow_zero):
        raise OasisException(f"{section} '{name}' must be a {'non-negative' if allow_zero else 'positive'} number, got {value!r}")


def _scheme(i, scheme, income_column):
    section = f"{SETTINGS_KEY}.premiums.schemes[{i}]"
    if isinstance(scheme, str):
        scheme = {'type': scheme}
    if not isinstance(scheme, dict) or scheme.get('type') not in PREMIUM_SCHEMES:
        raise OasisException(f"{section} must name a 'type' out of {list(PREMIUM_SCHEMES)}, got {scheme!r}")
    kind = scheme['type']
    _check_keys(section, scheme, {'type', 'name', *SCHEME_DEFAULTS[kind]})
    checked = {'name': scheme.get('name', kind), 'type': kind, **SCHEME_DEFAULTS[kind],
               **{k: v for k, v in scheme.items() if k not in ('type', 'name')}}
    if kind == 'flat' and checked['basis'] not in ('row', 'tiv'):
        raise OasisException(f"{section} 'basis' must be 'row' (the same premium for every row) or 'tiv' (the same rate "
                             f"per unit of value), got {checked['basis']!r}")
    if kind in ('capped', 'means_tested'):
        if income_column is None:
            raise OasisException(f"{section} ({kind}) needs {SETTINGS_KEY}.income_column")
        if checked['funding'] not in FUNDING:
            raise OasisException(f"{section} 'funding' must be one of {list(FUNDING)}, got {checked['funding']!r}")
    if kind == 'capped':
        _positive(section, 'cap', checked['cap'])
    if kind == 'means_tested':
        _positive(section, 'income_threshold', checked['income_threshold'])
        _positive(section, 'discount', checked['discount'])
        if checked['discount'] > 1:
            raise OasisException(f"{section} 'discount' is a share of the premium, between 0 and 1, got {checked['discount']!r}")
    return checked


def equity_report_settings(settings):
    """Return the ``equity_report`` settings with defaults filled in, or None if there are none.

    Args:
        settings (dict): Analysis settings.

    Returns:
        dict or None: The checked settings.

    Raises:
        OasisException: If a setting is unknown or has a bad value.
    """
    given = (settings or {}).get(SETTINGS_KEY)
    if not given:
        return None
    if not isinstance(given, dict):
        raise OasisException(f"{SETTINGS_KEY} must be an object, got {given!r}")
    _check_keys(SETTINGS_KEY, given, DEFAULTS)
    checked = {**DEFAULTS, **given}

    if checked['runtype'] not in RUNTYPES:
        raise OasisException(f"{SETTINGS_KEY} 'runtype' must be one of {list(RUNTYPES)}, got {checked['runtype']!r}")
    if isinstance(checked['group_by'], str):
        checked['group_by'] = [checked['group_by']]
    if not checked['group_by']:
        raise OasisException(f"{SETTINGS_KEY} needs 'group_by', the summary-info column(s) to compare groups by")
    if not isinstance(checked['summary_set'], int) or checked['summary_set'] < 1:
        raise OasisException(f"{SETTINGS_KEY} 'summary_set' must be a summary set id, got {checked['summary_set']!r}")
    for name in ('burden_threshold', 'income_burden_threshold'):
        _positive(SETTINGS_KEY, name, checked[name])
    if not isinstance(checked['min_group_size'], int) or checked['min_group_size'] < 0:
        raise OasisException(f"{SETTINGS_KEY} 'min_group_size' must be a whole number, 0 to report every group, "
                             f"got {checked['min_group_size']!r}")
    if checked['sample_type'] not in (None, 1, 2):
        raise OasisException(f"{SETTINGS_KEY} 'sample_type' must be 1 (analytical) or 2 (sample mean), got {checked['sample_type']!r}")

    weighting = checked['equity_weighting']
    if weighting:
        weighting = {} if weighting is True else weighting
        section = f"{SETTINGS_KEY}.equity_weighting"
        _check_keys(section, weighting, WEIGHTING_DEFAULTS)
        weighting = {**WEIGHTING_DEFAULTS, **weighting}
        if checked['income_column'] is None:
            raise OasisException(f"{section} needs {SETTINGS_KEY}.income_column")
        _positive(section, 'eta', weighting['eta'], allow_zero=True)
        if weighting['reference_income'] not in ('mean', 'median'):
            _positive(section, 'reference_income', weighting['reference_income'])
        periods = weighting['return_periods']
        if not periods or not all(isinstance(p, (int, float)) and p > 0 for p in periods):
            raise OasisException(f"{section} 'return_periods' must be a list of positive numbers, got {periods!r}")
        weighting['return_periods'] = sorted(set(periods))
        checked['equity_weighting'] = weighting
    else:
        checked['equity_weighting'] = None

    premiums = checked['premiums']
    if premiums:
        premiums = {} if premiums is True else premiums
        section = f"{SETTINGS_KEY}.premiums"
        _check_keys(section, premiums, PREMIUM_DEFAULTS)
        premiums = {**PREMIUM_DEFAULTS, **premiums}
        _positive(section, 'loading', premiums['loading'], allow_zero=True)
        schemes = premiums['schemes'] or ['risk_based', 'flat']
        premiums['schemes'] = [_scheme(i, s, checked['income_column']) for i, s in enumerate(schemes)]
        names = [s['name'] for s in premiums['schemes']]
        if len(set(names)) != len(names):
            raise OasisException(f"{section} schemes need different names, got {names}")
        checked['premiums'] = premiums
    else:
        checked['premiums'] = None

    gaps = checked['data_gaps']
    if gaps:
        gaps = {} if gaps is True else gaps
        section = f"{SETTINGS_KEY}.data_gaps"
        _check_keys(section, gaps, DATA_GAP_DEFAULTS)
        gaps = {**DATA_GAP_DEFAULTS, **gaps}
        ids = gaps['default_vulnerability_ids']
        if not isinstance(ids, list) or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
            raise OasisException(f"{section} 'default_vulnerability_ids' must be a list of vulnerability ids, got {ids!r}")
        if gaps['geocode_quality_threshold'] is not None:
            _positive(section, 'geocode_quality_threshold', gaps['geocode_quality_threshold'])
        checked['data_gaps'] = gaps
    else:
        checked['data_gaps'] = None

    sharing = checked['loss_sharing']
    if sharing:
        sharing = {} if sharing is True else sharing
        section = f"{SETTINGS_KEY}.loss_sharing"
        _check_keys(section, sharing, {'summary_sets'})
        sets = sharing.get('summary_sets') or {}
        if not isinstance(sets, dict):
            raise OasisException(f"{section} 'summary_sets' must map perspectives to summary set ids, got {sets!r}")
        _check_keys(f"{section}.summary_sets", sets, LOSS_SHARING_PERSPECTIVES)
        sets = {p: sets.get(p, checked['summary_set']) for p in LOSS_SHARING_PERSPECTIVES}
        for perspective, summary_set in sets.items():
            if not isinstance(summary_set, int) or isinstance(summary_set, bool) or summary_set < 1:
                raise OasisException(f"{section} summary set for '{perspective}' must be a summary set id, got {summary_set!r}")
        checked['loss_sharing'] = {'summary_sets': sets}
    else:
        checked['loss_sharing'] = None

    assistance = checked['public_assistance']
    if assistance:
        assistance = {} if assistance is True else assistance
        section = f"{SETTINGS_KEY}.public_assistance"
        _check_keys(section, assistance, ASSISTANCE_DEFAULTS)
        assistance = {**ASSISTANCE_DEFAULTS, **assistance}
        _positive(section, 'deductible', assistance['deductible'], allow_zero=True)
        _positive(section, 'share', assistance['share'])
        if assistance['share'] > 1:
            raise OasisException(f"{section} 'share' is the share of the retained loss paid, between 0 and 1, got {assistance['share']!r}")
        if assistance['limit'] is not None:
            _positive(section, 'limit', assistance['limit'])
        if assistance['income_threshold'] is not None:
            if checked['income_column'] is None:
                raise OasisException(f"{section} 'income_threshold' needs {SETTINGS_KEY}.income_column")
            _positive(section, 'income_threshold', assistance['income_threshold'])
        periods = assistance['return_periods']
        if not periods or not all(isinstance(p, (int, float)) and not isinstance(p, bool) and p > 0 for p in periods):
            raise OasisException(f"{section} 'return_periods' must be a list of positive numbers, got {periods!r}")
        assistance['return_periods'] = sorted(set(periods))
        checked['public_assistance'] = assistance
    else:
        checked['public_assistance'] = None

    basis = checked['basis_risk']
    if basis:
        basis = {} if basis is True else basis
        section = f"{SETTINGS_KEY}.basis_risk"
        _check_keys(section, basis, BASIS_RISK_DEFAULTS)
        basis = {**BASIS_RISK_DEFAULTS, **basis}
        for name in BASIS_RISK_DEFAULTS:
            _positive(section, name, basis[name], allow_zero=True)
        checked['basis_risk'] = basis
    else:
        checked['basis_risk'] = None
    return checked
