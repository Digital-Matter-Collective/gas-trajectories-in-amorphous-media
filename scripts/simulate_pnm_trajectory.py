import argparse
from pathlib import Path

from base.bufferedsampler import BufferedSampler
from base.discretecdf import DiscreteCDF
from base.reader import Reader
from processes.pnm_walk_simulator import PnmWalkSimulator, load_pnm_graph
from utils.logging_setup import setup_logging
from utils.utils import ps_generate
from visualizer.visualizer import Visualizer, WrapMode


def run(
    pnm_prefix: str,
    k: float = 0.5,
    p: float = 0.5,
    steps: int = 1000,
    radius: float = 0.02,
    min_radius: float = Reader.PNM_MIN_RADIUS_NM,
) -> None:
    ps_type = 'uniform'  # poisson uniform
    ps = ps_generate(ps_type, mean_count=50)
    bs_ps = BufferedSampler(DiscreteCDF(ps), "ps", size=100_000)

    graph = load_pnm_graph(
        pnm_prefix, scale=Reader.PNM_M_TO_NM, min_radius=min_radius
    )

    simulator = PnmWalkSimulator(graph, bs_ps, k, p, with_history=False)
    traj = simulator.run(steps)
    Visualizer.draw_trajectories(
        [traj],
        radius=radius,
        periodic=False,
        wrap_mode=WrapMode.EMPTY,
        with_points=True,
        color_type='dist',
    )
    Visualizer.show()


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description=(
            "Simulate and visualize one kerogen gas-molecule trajectory by "
            "walking a real pore-network model directly, instead of "
            "drawing trap size and inter-trap step length from fitted "
            "P(r)/P(h) distributions"
        )
    )
    parser.add_argument(
        "pnm_prefix",
        type=Path,
        help="PNM files prefix (without _node1.dat etc.)",
    )
    parser.add_argument(
        "--k", type=float, default=0.5, help="Trapping probability parameter"
    )
    parser.add_argument(
        "--p", type=float, default=0.5, help="Return probability parameter"
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="Number of simulated steps"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.02,
        help="Rendered trajectory tube/point radius",
    )
    parser.add_argument(
        "--min-radius",
        type=float,
        default=Reader.PNM_MIN_RADIUS_NM,
        help=(
            "Drop pores with radius <= this value before walking (default: "
            "0.003 nm; same convention as read_pnm_data)"
        ),
    )
    args = parser.parse_args()

    run(
        str(args.pnm_prefix),
        k=args.k,
        p=args.p,
        steps=args.steps,
        radius=args.radius,
        min_radius=args.min_radius,
    )


if __name__ == '__main__':
    main()
