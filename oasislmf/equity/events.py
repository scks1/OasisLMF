"""Read the ground-up and insured losses of each event, sample and row, matched between the two perspectives."""

__all__ = ['EVENT_KEYS', 'event_losses']

import logging

import pandas as pd

from ..utils.exceptions import OasisException
from .ord import ord_path, read_ord
from .sharing import perspective_rows, summary_id_map

logger = logging.getLogger(__name__)

EVENT_KEYS = ['Period', 'EventId', 'SampleId', 'SummaryId']


def _plt_losses(output_dir, runtype, summary_set, source, sample_type, what, id_map=None):
    if source == 'sample':
        plt = read_ord(output_dir, runtype, summary_set, 'splt', needed_for=f'{what} (plt_sample)')
        # the SPLT also holds the analytical mean as sample -1
        plt = plt[plt['SampleId'] > 0]
    else:
        plt = read_ord(output_dir, runtype, summary_set, 'mplt', needed_for=f'{what} (plt_moment)')
        plt = plt[plt['SampleType'] == sample_type].rename(columns={'MeanLoss': 'Loss'}).assign(SampleId=0)
    weights = plt.groupby('Period')['PeriodWeight'].first()
    if id_map is not None:
        plt = plt.assign(SummaryId=plt['SummaryId'].map(id_map)).dropna(subset=['SummaryId'])
        plt['SummaryId'] = plt['SummaryId'].astype('int64')
    return plt.groupby(EVENT_KEYS)['Loss'].sum(), weights


def event_losses(output_dir, settings, sample_type, analysis_settings, what, require_il, nonlinear_note):
    """Return the ground-up rows and the losses of each event, sample and row in the gul and il outputs.

    Losses come from the sample period loss tables (SPLT) when every perspective used has one and the analysis
    settings give ``number_of_samples``, otherwise from the mean losses of the moment period loss tables (MPLT).
    The il summary set's rows are matched to the ground-up ones by their summary-info values.

    Args:
        output_dir (str): Analysis output directory.
        settings (dict): Checked ``equity_report`` settings. The summary sets are those of ``loss_sharing``, or the
            report's ``summary_set``.
        sample_type (int): MPLT sample type when mean losses are used.
        analysis_settings (dict): Analysis settings, for ``number_of_samples``.
        what (str): What the losses are for, used in messages.
        require_il (bool): Raise if there are no il outputs. Otherwise, without them, ``il`` is None.
        nonlinear_note (str): Why mean losses fall short, logged when they are used.

    Returns:
        dict: ``rows`` (ground-up report rows), ``gul`` and ``il`` (Series of losses indexed by :data:`EVENT_KEYS`,
        with ground-up summary ids; ``il`` may be None), ``weights`` (weight of each period with a loss in either),
        ``samples`` (1 for mean losses) and ``source`` (``'sample'`` or ``'mean'``).

    Raises:
        OasisException: If outputs are missing or the il rows can't be matched to the ground-up rows.
    """
    sets = settings['loss_sharing']['summary_sets'] if settings['loss_sharing'] else {
        p: settings['summary_set'] for p in ('gul', 'il')}
    rows = perspective_rows(output_dir, settings, 'gul', sets['gul'], sample_type)
    try:
        il_rows = perspective_rows(output_dir, settings, 'il', sets['il'], sample_type)
    except OasisException:
        if require_il:
            raise
        il_rows = None
        logger.info("%s: no il_S%s outputs, so every ground-up loss counts as retained", what, sets['il'])
    id_map = None
    if il_rows is not None:
        id_map = summary_id_map(rows, il_rows)
        if id_map is None:
            raise OasisException(
                f"{what} needs the gul and il summary sets to have the same rows (for example, both grouped by "
                "LocNumber), so each row's losses can be compared")

    samples = (analysis_settings or {}).get('number_of_samples')
    have_splt = ord_path(output_dir, 'gul', sets['gul'], 'splt') and (
        il_rows is None or ord_path(output_dir, 'il', sets['il'], 'splt'))
    if have_splt and isinstance(samples, int) and not isinstance(samples, bool) and samples > 0:
        source = 'sample'
    else:
        source, samples = 'mean', 1
        logger.info("%s: using each event's mean loss (MPLT), %s; turn on plt_sample for every perspective used, with "
                    "number_of_samples set, to use each sample", what, nonlinear_note)

    gul, weights = _plt_losses(output_dir, 'gul', sets['gul'], source, sample_type, what)
    il = None
    if il_rows is not None:
        il, il_weights = _plt_losses(output_dir, 'il', sets['il'], source, sample_type, what, id_map)
        weights = pd.concat([weights, il_weights]).groupby(level=0).first()
    return {'rows': rows, 'gul': gul, 'il': il, 'weights': weights, 'samples': samples, 'source': source}
