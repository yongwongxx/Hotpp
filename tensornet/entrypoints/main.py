import argparse, importlib, logging
from .. import __version__, __picture__
from ..logger import set_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="MiaoNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--version",
        help="print version",
        action='version', 
        version=__version__
    )
    parser_log = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_log.add_argument(
        "-ll",
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="set verbosity level by strings: ERROR, WARNING, INFO and DEBUG",
    )
    parser_log.add_argument(
        "-lp",
        "--log-path",
        type=str,
        default="log.txt",
        help="set log file to log messages to disk",
    )
    subparsers = parser.add_subparsers(title="Valid subcommands", dest="command")
    # train
    parser_train = subparsers.add_parser(
        "train",
        parents=[parser_log],
        help="train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_train.add_argument(
        "-i",
        "--input-file",
        type=str, 
        default="input.yaml",
        help="the input parameter file in yaml format"
    )
    parser_train.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="Load model from model directly.",
    )
    parser_train.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="Load model, optimizer and lr_scheduler statedict from checkpoint.",
    )
    # eval
    parser_eval = subparsers.add_parser(
        "eval",
        help="train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_eval.add_argument(
        "-m",
        "--model",
        type=str, 
        default="model.pt",
        help="model"
    )
    parser_eval.add_argument(
        "-c",
        "--cutoff",
        type=float, 
        default=None,
        help="cutoff"
    )
    parser_eval.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cpu",
        help="device"
    )
    parser_eval.add_argument(
        "-d",
        "--dataset",
        type=str, 
        default="data.traj",
        help="dataset"
    )
    parser_eval.add_argument(
        "-f",
        "--format",
        type=str, 
        default=None,
        help="format"
    )
    parser_eval.add_argument(
        "-b",
        "--batchsize",
        type=int, 
        default=32,
        help="batchsize"
    )
    parser_eval.add_argument(
        "-p",
        "--properties",
        type=str,
        nargs="+",
        default=["energy", "forces"],
        help="target properties"
    )
    # clean
    parser_clean = subparsers.add_parser(
        "clean",
        help="clean",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # plot
    parser_plot = subparsers.add_parser(
        "plot",
        help="plot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_plot.add_argument(
        "-p",
        "--properties",
        type=str,
        nargs="+",
        default=["per_energy", "forces"],
        help="target properties"
    )
    # freeze
    parser_freeze = subparsers.add_parser(
        "freeze",
        help="freeze",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_freeze.add_argument(
        "-m",
        "--model",
        default="model.pt",
        help="model",
    )
    parser_freeze.add_argument(
        "-s",
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="symbols"
    )
    parser_freeze.add_argument(
        "-d",
        "--device",
        choices=["cuda", "cpu"],
        default="cpu",
        help="device"
    )
    parser_freeze.add_argument(
        "-o",
        "--output",
        default="infer.pt",
        help="output",
    )
    parsed_args = parser.parse_args()
    if parsed_args.command is None:
        print(__picture__)
        parser.print_help()
    return parsed_args


def main():
    args = parse_args()
    dict_args = vars(args)
    if args.command in ['train']:
        set_logger(level=dict_args['log_level'], log_path=dict_args['log_path'])
        log = logging.getLogger(__name__)
        log.info(__picture__)
    if args.command:
        try:
            f = getattr(importlib.import_module('tensornet.entrypoints.{}'.format(args.command)), "main")
        except:
            raise RuntimeError(f"unknown command {args.command}")
        f(**dict_args)

if __name__ == "__main__":
    main()
