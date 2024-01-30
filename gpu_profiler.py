from metaflow.cards import Markdown, Table, VegaChart
from metaflow.plugins.cards.card_modules.card import MetaflowCardComponent
from metaflow.plugins.cards.card_modules.components import with_default_component_id
from functools import wraps
import threading
from datetime import datetime
from metaflow import current
from metaflow.cards import Table, Markdown, VegaChart, Image
import time

import re
import os
from tempfile import TemporaryFile
from subprocess import check_output, Popen
from datetime import datetime
from functools import wraps

# Card plot styles
MEM_COLOR = "#0c64d6"
GPU_COLOR = "#ff69b4"


DRIVER_VER = re.compile(b"Driver Version: (.+?) ")
CUDA_VER = re.compile(b"CUDA Version:(.*) ")

MONITOR_FIELDS = [
    "timestamp",
    "gpu_utilization",
    "memory_used",
    "memory_total",
]

MONITOR = """
set -e;
while kill -0 {pid} 2>/dev/null;
do
 nvidia-smi
   --query-gpu=pci.bus_id,timestamp,utilization.gpu,memory.used,memory.total
   --format=csv,noheader,nounits;
 sleep {interval};
done
""".replace(
    "\n", " "
)


def _update_utilization(results, md_dict):
    for device, data in results["profile"].items():
        md_dict[device]["gpu"].update(
            "%2.1f%%" % max(map(float, data["gpu_utilization"]))
        )
        md_dict[device]["memory"].update("%dMB" % max(map(float, data["memory_used"])))


def _update_charts(results, md_dict):
    for device, data in results["profile"].items():
        gpu_plot, mem_plot = profile_plots(
            device,
            data["timestamp"],
            data["gpu_utilization"],
            data["memory_used"],
            data["memory_total"],
        )
        md_dict[device]["gpu"].update(gpu_plot)
        md_dict[device]["memory"].update(mem_plot)


# This code is adapted from: https://github.com/outerbounds/monitorbench
class GPUProfiler:
    def __init__(self, interval=1):
        self.driver_ver, self.cuda_ver, self.error = self._read_versions()
        (
            self.interconnect_data,
            self.interconnect_legend,
        ) = self._read_multi_gpu_interconnect()
        if self.error:
            self.devices = []
            return
        else:
            self.devices = self._read_devices()
            self._monitor_out = TemporaryFile()
            cmd = MONITOR.format(interval=interval, pid=os.getpid())
            self._interval = interval
            self._monitor_proc = Popen(["bash", "-c", cmd], stdout=self._monitor_out)

        self._card_comps = {"max_utilization": {}, "charts": {}}
        self._card_created = False

    def finish(self):
        ret = {
            "error": self.error,
            "cuda_version": self.cuda_ver,
            "driver_version": self.driver_ver,
        }
        if self.error:
            return ret
        else:
            self._monitor_proc.terminate()
            ret["devices"] = self.devices
            ret["profile"] = self._read_monitor()
            ret["interconnect"] = {
                "data": self.interconnect_data,
                "legend": self.interconnect_legend,
            }
            return ret

    def _make_reading(self):
        ret = {
            "error": self.error,
            "cuda_version": self.cuda_ver,
            "driver_version": self.driver_ver,
        }
        if self.error:
            return ret
        else:
            ret["devices"] = self.devices
            ret["profile"] = self._read_monitor()
            ret["interconnect"] = {
                "data": self.interconnect_data,
                "legend": self.interconnect_legend,
            }
            return ret

    def _update_card(self):
        if len(self.devices) == 0:

            current.card["gpu_profile"].clear()
            current.card["gpu_profile"].append(
                Markdown("## GPU profile failed: %s" % self.error)
            )
            current.card["gpu_profile"].refresh()

            return

        while True:
            readings = self._make_reading()
            if readings is None:
                time.sleep(self._interval)
                continue
            _update_utilization(readings, self._card_comps["max_utilization"])
            _update_charts(readings, self._card_comps["charts"])
            current.card["gpu_profile"].refresh()
            time.sleep(self._interval)

    def _setup_card(self, artifact_name):
        from metaflow import current

        results = self._make_reading()
        els = current.card["gpu_profile"]

        def _drivers():
            els.append(Markdown("## Drivers"))
            els.append(
                Table(
                    [[results["cuda_version"], results["driver_version"]]],
                    headers=["NVidia driver version", "CUDA version"],
                )
            )

        def _devices():
            els.append(Markdown("## Devices"))
            rows = [
                [d["device_id"], d["name"], d["memory"]] for d in results["devices"]
            ]
            els.append(Table(rows, headers=["Device ID", "Device type", "GPU memory"]))

        def _interconnect():
            if results["interconnect"]["data"] and results["interconnect"]["legend"]:
                els.append(Markdown("## Interconnect"))
                interconnect_data = results["interconnect"]["data"]
                rows = list(interconnect_data.values())
                rows = [list(transpose_row) for transpose_row in list(zip(*rows))]
                els.append(Table(rows, headers=list(interconnect_data.keys())))
                els.append(Markdown("#### Legend"))
                els.append(
                    Table(
                        [list(results["interconnect"]["legend"].values())],
                        headers=list(results["interconnect"]["legend"].keys()),
                    )
                )

        def _utilization():
            els.append(Markdown("## Maximum utilization"))
            rows = {}
            for d in results["devices"]:
                rows[d["device_id"]] = {
                    "gpu": Markdown("0%"),
                    "memory": Markdown("0MB"),
                }
            _rows = [[Markdown(k)] + list(v.values()) for k, v in rows.items()]
            els.append(
                Table(data=_rows, headers=["Device ID", "Max GPU %", "Max memory"])
            )
            els.append(
                Markdown(f"Detailed data saved in an artifact `{artifact_name}`")
            )
            return rows

        def _plots():
            els.append(Markdown("## GPU utilization and memory usage over time"))

            rows = {}
            for d in results["devices"]:
                gpu_plot, mem_plot = profile_plots(d["device_id"], [], [], [], [])
                rows[d["device_id"]] = {
                    "gpu": VegaChart(gpu_plot),
                    "memory": VegaChart(mem_plot),
                }
            for k, v in rows.items():
                els.append(Markdown("### GPU Utilization for device : %s" % k))
                els.append(
                    Table(
                        data=[
                            [Markdown("GPU Utilization"), v["gpu"]],
                            [Markdown("Memory usage"), v["memory"]],
                        ]
                    )
                )
            return rows

        _drivers()
        _devices()
        _interconnect()
        self._card_comps["max_utilization"] = _utilization()
        self._card_comps["charts"] = _plots()

    def _read_monitor(self):
        devdata = {}
        self._monitor_out.seek(0)
        for line in self._monitor_out:
            fields = [f.strip() for f in line.decode("utf-8").split(",")]
            if len(fields) == len(MONITOR_FIELDS) + 1:
                # strip subsecond resolution from timestamps that doesn't align across devices
                fields[1] = fields[1].split(".")[0]
                if fields[0] in devdata:
                    data = devdata[fields[0]]
                else:
                    devdata[fields[0]] = data = {}

                for i, field in enumerate(MONITOR_FIELDS):
                    if field not in data:
                        data[field] = []
                    data[field].append(fields[i + 1])
            else:
                # expect that the last line may be truncated
                break
        return devdata

    def _read_versions(self):
        def parse(r, s):
            return r.search(s).group(1).strip().decode("utf-8")

        try:
            out = check_output(["nvidia-smi"])
            return parse(DRIVER_VER, out), parse(CUDA_VER, out), None
        except FileNotFoundError:
            return None, None, "nvidia-smi not found"
        except AttributeError:
            return None, None, "nvidia-smi output is unexpected"
        except:
            return None, None, "nvidia-smi error"

    def _read_devices(self):
        out = check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,pci.bus_id,memory.total",
                "--format=csv,noheader",
            ]
        )
        return [
            dict(
                zip(("name", "device_id", "memory"), (x.strip() for x in l.split(",")))
            )
            for l in out.decode("utf-8").splitlines()
        ]

    def _read_multi_gpu_interconnect(self):
        """
        parse output of `nvidia-smi tomo -m`, such as this sample:

            GPU0    GPU1    CPU Affinity    NUMA Affinity
            GPU0     X      NV2     0-23            N/A
            GPU1    NV2      X      0-23            N/A

        returns two dictionaries describing multi-GPU topology:
            data: {index: [GPU0, GPU1, ...], GPU0: [X, NV2, ...], GPU1: [NV2, X, ...], ...}
            legend_items: {X: 'Same PCI', NV2: 'NVLink 2', ...}
        """
        try:

            import re

            ansi_escape = re.compile(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]")

            out = check_output(["nvidia-smi", "topo", "-m"])
            rows = out.decode("utf-8").split("\n")

            header = ansi_escape.sub("", rows[0]).split("\t")[1:]
            data = {}
            data["index"] = []
            data |= {k: [] for k in header}

            for i, row in enumerate(rows[1:]):
                row = ansi_escape.sub("", row).split()
                if len(row) == 0:
                    continue
                if row[0].startswith("GPU"):
                    data["index"].append(row[0])
                    for key, val in zip(header, row[1:]):
                        data[key].append(val)
                elif row[0].startswith("Legend"):
                    break

            legend_items = {}
            for legend_row in rows[i:]:
                if legend_row == "" or legend_row.startswith("Legend"):
                    continue
                res = legend_row.strip().split(" = ")
                legend_items[res[0].strip()] = res[1].strip()

            return data, legend_items

        except:
            return None, None


class gpu_profile:
    def __init__(
        self,
        include_artifacts=True,
        artifact_prefix="gpu_profile_",
        interval=1,
    ):
        self.include_artifacts = include_artifacts
        self.artifact_prefix = artifact_prefix
        self.interval = interval

    def __call__(self, f):
        @wraps(f)
        def func(s):
            prof = GPUProfiler(interval=self.interval)
            if self.include_artifacts:
                setattr(s, self.artifact_prefix + "num_gpus", len(prof.devices))

            current.card["gpu_profile"].append(
                Markdown("# GPU profile for `%s`" % current.pathspec)
            )
            prof._setup_card(self.artifact_prefix + "data")
            current.card["gpu_profile"].refresh()
            update_thread = threading.Thread(target=prof._update_card, daemon=True)
            update_thread.start()

            try:
                f(s)
            finally:
                try:
                    results = prof.finish()
                except:
                    results = {"error": "couldn't read profiler results"}
                if self.include_artifacts:
                    setattr(s, self.artifact_prefix + "data", results)

        from metaflow import card

        return card(type="blank", id="gpu_profile", refresh_interval=self.interval)(
            func
        )


def translate_to_vegalite(
    tstamps, vals, description ,y_label, legend, line_color=None, percentage_format=False
):
    # Preprocessing for Vega-Lite
    # Assuming tstamps is a list of datetime objects and vals is a list of values
    data = [{"tstamps": str(t), "vals": v} for t, v in zip(tstamps, vals)]

    # Base Vega-Lite spec
    vega_lite_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": description,
        "data": {"values": data},
        "width": 600,
        "height": 400,
        "encoding": {
            "x": {"field": "tstamps", "type": "temporal", "axis": {"title": "Time"}},
            "y": {
                "field": "vals",
                "type": "quantitative",
                "axis": {
                    "title": y_label,
                    **({"format": "%"} if percentage_format else {}),
                },
            },
        },
        "layer": [
            {
                "mark": {
                    "type": "line",
                    "color": line_color if line_color else "blue",
                    "tooltip": True,
                    "description": legend,  # Adding legend as description
                },
                "encoding": {"tooltip": [{"field": "tstamps"}, {"field": "vals"}]},
            }
        ],
    }

    return vega_lite_spec


def profile_plots(device_id, ts, gpu, mem_used, mem_total):
    tstamps = [datetime.strptime(t, "%Y/%m/%d %H:%M:%S") for t in ts]
    gpu = [i / 100 for i in list(map(float, gpu))]
    mem = [float(used) / float(total) for used, total in zip(mem_used, mem_total)]

    gpu_plot = translate_to_vegalite(
        tstamps,
        gpu,
        "GPU utilization",
        "GPU utilization",
        "device: %s" % device_id,
        line_color=GPU_COLOR,
        percentage_format=True,
    )
    mem_plot = translate_to_vegalite(
        tstamps,
        mem,
        "Percentage Memory utilization",
        "Percentage Memory utilization",
        "device: %s" % device_id,
        line_color=MEM_COLOR,
        percentage_format=True,
    )
    return gpu_plot, mem_plot
