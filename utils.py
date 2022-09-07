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

