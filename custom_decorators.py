from functools import wraps


def _install_with_pip(file=None, libraries=None):
    import os
    import subprocess
    import sys

    _libraries = {}
    if file is not None:
        with open(file, "r") as reqs:
            lines = [line.split("\n")[0] for line in reqs.readlines()]
            for line in lines:
                result = line.split("==")
                if len(result) == 2:
                    library, version = result[0], result[1]
                    _libraries[library] = version
                elif len(result) == 1:
                    library = result[0]
                    _libraries[library] = ""
                else:
                    raise ValueError("Each line in requirements.txt file ")

    else:
        _libraries = libraries

    for library, version in _libraries.items():
        print("Pip Install:", library, version)
        if version != "":
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    library + "==" + version,
                ]
            )
        else:
            subprocess.run([sys.executable, "-m", "pip", "install", library])


def _try_loading_matplotlib():
    try:
        import matplotlib
    except ImportError:
        _install_with_pip(libraries={"matplotlib": "3.5.3"})


def pip(file=None, libraries=None):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            _install_with_pip(file=file, libraries=libraries)
            return function(*args, **kwargs)

        return wrapper

    return decorator


def enable_decorator(dec, flag):
    flag = int(flag)
    assert flag in [
        0,
        1,
    ], "Flag must be set to a 0 or 1. Set it in CLI like: `export REMOTE=1`"

    def decorator(func):
        if flag:
            return dec(func)
        return func

    return decorator


import re
import os
from tempfile import TemporaryFile
from subprocess import check_call, check_output, Popen
from datetime import datetime
from functools import wraps

# Card plot styles
MEM_COLOR = "#0c64d6"
GPU_COLOR = "#ff69b4"
AXES_COLOR = "#666"
LABEL_COLOR = "#333"
FONTSIZE = 10
AXES_LINEWIDTH = 0.5
PLOT_LINEWIDTH = 1.2
WIDTH = 12
HEIGHT = 8

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


class GPUProfiler:
    def __init__(self, interval=1):
        self.driver_ver, self.cuda_ver, self.error = self._read_versions()
        if self.error:
            self.devices = []
        else:
            self.devices = self._read_devices()
            self._monitor_out = TemporaryFile()
            cmd = MONITOR.format(interval=interval, pid=os.getpid())
            self._monitor_proc = Popen(["bash", "-c", cmd], stdout=self._monitor_out)

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
            return ret

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


class gpu_profile:
    def __init__(
        self,
        with_card=True,
        include_artifacts=True,
        artifact_prefix="gpu_profile_",
        interval=1,
    ):
        self.with_card = with_card
        self.include_artifacts = include_artifacts
        self.artifact_prefix = artifact_prefix
        self.interval = interval

    def __call__(self, f):
        @wraps(f)
        def func(s):
            prof = GPUProfiler(interval=self.interval)
            if self.include_artifacts:
                setattr(s, self.artifact_prefix + "num_gpus", len(prof.devices))
            try:
                f(s)
            finally:
                try:
                    results = prof.finish()
                except:
                    results = {"error": "couldn't read profiler results"}
                if self.include_artifacts:
                    setattr(s, self.artifact_prefix + "data", results)
                if self.with_card:
                    try:
                        make_card(results, self.artifact_prefix + "data")
                    except:
                        pass

        if self.with_card:
            from metaflow import card

            return card(type="blank", id="gpu_profile")(func)
        else:
            return func


def make_plot(
    tstamps,
    vals,
    y_label,
    legend,
    line_color=None,
    secondary_y_factor=None,
    secondary_y_label="",
):
    import matplotlib.dates as mdates
    import matplotlib.ticker as mtick
    import matplotlib.pyplot as plt

    first = tstamps[0]

    def seconds_since_start(x):
        return (x - mdates.date2num(first)) * (24 * 60 * 60)

    with plt.rc_context(
        {
            "axes.edgecolor": AXES_COLOR,
            "axes.linewidth": AXES_LINEWIDTH,
            "xtick.color": AXES_COLOR,
            "ytick.color": AXES_COLOR,
            "text.color": LABEL_COLOR,
            "font.size": FONTSIZE,
        }
    ):
        fig = plt.figure(figsize=(WIDTH, HEIGHT))
        ax = fig.add_subplot(111)

        # left Y axis shows %
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylabel(y_label, labelpad=20)

        # top X axis shows seconds since start
        topax = ax.secondary_xaxis("top", functions=(seconds_since_start, lambda _: _))
        topax.set_xlabel("Seconds since task start", labelpad=20)
        # strange bug - secondary x axis become slightly thicker without this
        topax.spines["top"].set_linewidth(0)

        # bottom X axis shows timestamp
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
        )
        ax.set_xlabel("Time", labelpad=20)

        if secondary_y_factor is not None:
            rightax = ax.secondary_yaxis(
                "right",
                functions=(
                    lambda x: (x / 100) * secondary_y_factor,
                    lambda x: (x / 100) / secondary_y_factor,
                ),
            )
            rightax.set_ylabel(secondary_y_label, labelpad=20)

        line = ax.plot(tstamps, vals, linewidth=PLOT_LINEWIDTH, color=line_color)
        ax.legend([legend], loc="upper left")
    return ax


def profile_plots(device_id, profile_data):
    data = profile_data[device_id]
    tstamps = [datetime.strptime(t, "%Y/%m/%d %H:%M:%S") for t in data["timestamp"]]
    gpu = list(map(float, data["gpu_utilization"]))
    mem = [
        100.0 * float(used) / float(total)
        for used, total in zip(data["memory_used"], data["memory_total"])
    ]
    gpu_plot = make_plot(
        tstamps, gpu, "GPU utilization", "device: %s" % device_id, line_color=GPU_COLOR
    )
    mem_plot = make_plot(
        tstamps,
        mem,
        "Memory utilization",
        "device: %s" % device_id,
        line_color=MEM_COLOR,
        secondary_y_factor=float(data["memory_total"][0]),
        secondary_y_label="Memory usage in MBs",
    )
    return gpu_plot, mem_plot


def make_card(results, artifact_name):
    from metaflow import current
    from metaflow.cards import Table, Markdown, Image

    els = []

    def _error():
        els.append(Markdown(f"## GPU profiler failed:\n```results['error']```"))

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
        rows = [[d["device_id"], d["name"], d["memory"]] for d in results["devices"]]
        els.append(Table(rows, headers=["Device ID", "Device type", "GPU memory"]))

    def _utilization():
        els.append(Markdown("## Maximum utilization"))
        rows = []
        for device, data in results["profile"].items():
            max_gpu = max(map(float, data["gpu_utilization"]))
            max_mem = max(map(float, data["memory_used"]))
            rows.append([device, "%2.1f%%" % max_gpu, "%dMB" % max_mem])
        els.append(Table(rows, headers=["Device ID", "Max GPU %", "Max memory"]))
        els.append(Markdown(f"Detailed data saved in an artifact `{artifact_name}`"))

    def _plots():
        rows = []
        for device in results["profile"]:
            gpu_plot, mem_plot = profile_plots(device, results["profile"])
            rows.append(
                [
                    device,
                    Image.from_matplotlib(gpu_plot),
                    Image.from_matplotlib(mem_plot),
                ]
            )
        els.append(
            Table(rows, headers=["Device ID", "GPU Utilization", "Memory usage"])
        )

    els.append(Markdown(f"# GPU profile for `{current.pathspec}`"))
    if results["error"]:
        _error()
    else:
        _drivers()
        _devices()
        _utilization()

        _try_loading_matplotlib()
        try:
            import matplotlib
        except:
            els.append(Markdown("Install `matplotlib` to enable plots"))
        else:
            try:
                _plots()
            except:
                els.append(Markdown("Couldn't create plots"))

    for el in els:
        current.card["gpu_profile"].append(el)
