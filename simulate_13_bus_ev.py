from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Line:
    parent: int
    child: int
    r_pu: float
    x_pu: float


@dataclass(frozen=True)
class EVCharger:
    bus: int
    count: int
    kw_per_vehicle: float
    kvar_per_vehicle: float = 0.0


# Simple radial representation for a 13-bus feeder.
LINES: List[Line] = [
    Line(1, 2, 0.008, 0.010),
    Line(2, 3, 0.012, 0.014),
    Line(3, 4, 0.010, 0.013),
    Line(4, 5, 0.009, 0.012),
    Line(5, 6, 0.011, 0.014),
    Line(3, 7, 0.013, 0.016),
    Line(7, 8, 0.010, 0.012),
    Line(8, 9, 0.009, 0.011),
    Line(2, 10, 0.014, 0.017),
    Line(10, 11, 0.010, 0.013),
    Line(11, 12, 0.008, 0.010),
    Line(12, 13, 0.010, 0.012),
]


# Base active/reactive loads (kW/kVAr) at each bus.
BASE_LOAD_KW: Dict[int, Tuple[float, float]] = {
    2: (120.0, 70.0),
    3: (90.0, 52.0),
    4: (60.0, 35.0),
    5: (50.0, 28.0),
    6: (45.0, 25.0),
    7: (75.0, 42.0),
    8: (40.0, 24.0),
    9: (30.0, 18.0),
    10: (65.0, 38.0),
    11: (55.0, 30.0),
    12: (45.0, 26.0),
    13: (35.0, 20.0),
}


DEFAULT_EV_SCENARIO: List[EVCharger] = [
    EVCharger(bus=4, count=20, kw_per_vehicle=7.2, kvar_per_vehicle=1.4),
    EVCharger(bus=8, count=15, kw_per_vehicle=7.2, kvar_per_vehicle=1.4),
    EVCharger(bus=12, count=25, kw_per_vehicle=7.2, kvar_per_vehicle=1.4),
]


def _children_map(lines: List[Line]) -> Dict[int, List[int]]:
    children: Dict[int, List[int]] = {}
    for line in lines:
        children.setdefault(line.parent, []).append(line.child)
    return children


def _parent_map(lines: List[Line]) -> Dict[int, int]:
    return {line.child: line.parent for line in lines}


def _line_map(lines: List[Line]) -> Dict[Tuple[int, int], Line]:
    return {(line.parent, line.child): line for line in lines}


def _bus_order(lines: List[Line]) -> List[int]:
    buses = {1}
    for line in lines:
        buses.add(line.parent)
        buses.add(line.child)
    return sorted(buses)


def _bus_depths(children: Dict[int, List[int]], root: int = 1) -> Dict[int, int]:
    stack = [(root, 0)]
    depths: Dict[int, int] = {}
    while stack:
        node, depth = stack.pop()
        depths[node] = depth
        for child in children.get(node, []):
            stack.append((child, depth + 1))
    return depths


def _build_loads(ev_scenario: List[EVCharger]) -> Dict[int, Tuple[float, float]]:
    loads = dict(BASE_LOAD_KW)
    for charger in ev_scenario:
        base_p, base_q = loads.get(charger.bus, (0.0, 0.0))
        loads[charger.bus] = (
            base_p + charger.count * charger.kw_per_vehicle,
            base_q + charger.count * charger.kvar_per_vehicle,
        )
    return loads


def run_power_flow(
    ev_scenario: List[EVCharger] | None = None,
    base_power_kva: float = 1000.0,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
) -> Dict[str, object]:
    ev_scenario = ev_scenario or DEFAULT_EV_SCENARIO
    loads_kw = _build_loads(ev_scenario)
    children = _children_map(LINES)
    parents = _parent_map(LINES)
    line_lookup = _line_map(LINES)
    buses = _bus_order(LINES)
    depths = _bus_depths(children, 1)
    descending = sorted((bus for bus in buses if bus != 1), key=lambda b: depths[b], reverse=True)
    ascending = sorted((bus for bus in buses if bus != 1), key=lambda b: depths[b])

    # Initial voltage profile.
    voltages = {bus: complex(1.0, 0.0) for bus in buses}
    currents: Dict[Tuple[int, int], complex] = {(line.parent, line.child): 0j for line in LINES}

    load_pu = {bus: complex(p / base_power_kva, q / base_power_kva) for bus, (p, q) in loads_kw.items()}

    for _ in range(max_iterations):
        old = voltages.copy()
        injected_current: Dict[int, complex] = {}
        for bus in buses:
            s = load_pu.get(bus, 0j)
            v = voltages[bus]
            injected_current[bus] = 0j if bus == 1 or abs(v) < 1e-12 else (s / v).conjugate()

        branch_current_by_bus: Dict[int, complex] = {bus: injected_current.get(bus, 0j) for bus in buses}
        for bus in descending:
            parent = parents[bus]
            i_branch = branch_current_by_bus[bus]
            currents[(parent, bus)] = i_branch
            branch_current_by_bus[parent] = branch_current_by_bus.get(parent, 0j) + i_branch

        voltages[1] = complex(1.0, 0.0)
        for bus in ascending:
            parent = parents[bus]
            line = line_lookup[(parent, bus)]
            z = complex(line.r_pu, line.x_pu)
            voltages[bus] = voltages[parent] - z * currents[(parent, bus)]

        delta = max(abs(voltages[b] - old[b]) for b in buses)
        if delta < tolerance:
            break

    line_losses_kw = {}
    total_losses_kw = 0.0
    for line in LINES:
        i = currents[(line.parent, line.child)]
        loss_pu = (abs(i) ** 2) * line.r_pu
        loss_kw = loss_pu * base_power_kva
        line_losses_kw[(line.parent, line.child)] = loss_kw
        total_losses_kw += loss_kw

    voltage_magnitudes = {bus: abs(v) for bus, v in voltages.items()}
    total_kw = sum(p for p, _ in loads_kw.values())
    total_kvar = sum(q for _, q in loads_kw.values())

    return {
        "voltages_pu": voltage_magnitudes,
        "line_losses_kw": line_losses_kw,
        "total_load_kw": total_kw,
        "total_load_kvar": total_kvar,
        "total_losses_kw": total_losses_kw,
    }


def _format_results(results: Dict[str, object]) -> str:
    voltage_rows = "\n".join(
        f"  Bus {bus:>2}: {v:.4f} pu"
        for bus, v in sorted(results["voltages_pu"].items())  # type: ignore[index]
    )
    min_v = min(results["voltages_pu"].values())  # type: ignore[union-attr]
    max_v = max(results["voltages_pu"].values())  # type: ignore[union-attr]
    summary = [
        "13-Bus EV Integration Power Flow Results",
        f"Total real load : {results['total_load_kw']:.2f} kW",
        f"Total reactive  : {results['total_load_kvar']:.2f} kVAr",
        f"Total line loss : {results['total_losses_kw']:.2f} kW",
        f"Voltage range   : {min_v:.4f} to {max_v:.4f} pu",
        "",
        "Bus Voltages:",
        voltage_rows,
    ]
    return "\n".join(summary)


if __name__ == "__main__":
    output = run_power_flow()
    print(_format_results(output))
