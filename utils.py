import math
from metaflow.cards import get_cards


def create_chunk_ranges(arr, chunk_size):
    num_splits = math.ceil(len(arr) / chunk_size)
    index_list = []  # Will hold start,end delimiters
    for i in range(1, num_splits + 1):
        end = i * chunk_size
        start = (i - 1) * chunk_size
        if end > len(arr):
            end = start + (len(arr) - start)
        index_list.append((start, end))
    return index_list


def create_card_url(ui_url, task):
    fl, rn, st, ts = task.pathspec.split("/")
    cards = get_cards(task)
    return "%s/api/flows/%s/runs/%s/steps/%s/tasks/%s/cards/%s" % (
        ui_url,
        fl,
        rn,
        st,
        ts,
        cards[0].hash,
    )


def create_prompt(prompt, style):
    return "%s by %s" % (prompt, style)


def create_card_index(metaflow_ui_url=None, cols=3):
    """construct the URL of each card in one single index and add it to current card"""
    card_paths = []
    if metaflow_ui_url is None:
        return
    mf_ui_url = metaflow_ui_url
    if metaflow_ui_url.endswith("/"):
        mf_ui_url = metaflow_ui_url[:-1]

    from metaflow import current, Run, parallel_map, Task
    from metaflow.cards import Markdown, Table
    from collections import defaultdict

    run_pathspec = "/".join(([current.flow_name, current.run_id]))
    tasks_pathspecs = [t.pathspec for t in list(Run(run_pathspec)["paint_cards"])]

    def make_md_str(pthspc):
        t = Task(pthspc)
        style = t["inference_style"].data
        prompts = ", ".join(set([p for p, _, _ in t["image_index"].data]))
        url_path = create_card_url(mf_ui_url, t)
        md_str = "## [%s](%s)" % (prompts, url_path)
        return md_str, style

    md_strs = parallel_map(make_md_str, tasks_pathspecs)

    tab_res = defaultdict(list)
    for md_str, style in md_strs:
        tab_res[style].append(md_str)
    for k, v in tab_res.items():
        card_paths.append([Markdown("## %s" % k)] + [Markdown(vx) for vx in v])
    return [Markdown("# Path To Cards On Metaflow UI"), Table(card_paths)]


def unit_convert(number, base_unit, convert_unit):
    # base_unit : GB or MB or KB or B
    # convert_unit : GB or MB or KB or B
    # number : number of base_unit
    # return : number of convert_unit
    units = ["B", "KB", "MB", "GB"]
    if base_unit not in units or convert_unit not in units:
        raise ValueError("Invalid unit")
    base_unit_index = units.index(base_unit)
    convert_unit_index = units.index(convert_unit)
    factor = pow(1024, abs(base_unit_index - convert_unit_index))
    if base_unit_index < convert_unit_index:
        return round(number / factor, 3)
    else:
        return round(number * factor, 3)
