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
    parser_search = subparsers.add_parser(
        "train",
        parents=[parser_log],
        help="train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_search.add_argument(
        "-i",
        "--input-file",
        type=str, 
        default="input.yaml",
        help="the input parameter file in yaml format"
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
