import os

import nude
from argv.checkpoints import arg_checkpoints, set_arg_checkpoints, check_arg_checkpoints
from argv.common import arg_debug, arg_help, arg_version
from argv.run import arg_json_folder_name, arg_json_args, arg_gpu, arg_cpu, arg_preferences, \
    arg_color_transfer, arg_compress, arg_image_size, arg_ignore_size, arg_auto_resize_crop, arg_auto_resize, \
    arg_auto_rescale, arg_n_core
from argv.run.config import set_arg_preference, set_gpu_ids


def init_nude_sub_parser(subparsers):
    nude_parser = subparsers.add_parser(
        'nude',
        description="Running Nudifier on nude mode.",
        help="Running Nudifier on nude mode.",
        add_help=False
    )
    nude_parser.set_defaults(func=nude.main)

    # conflicts handler
    processing_mod = nude_parser.add_mutually_exclusive_group()
    scale_mod = nude_parser.add_mutually_exclusive_group()

    # add nude arguments
    arg_input(nude_parser)
    arg_output(nude_parser)

    arg_auto_rescale(scale_mod)
    arg_auto_resize(scale_mod)
    arg_auto_resize_crop(scale_mod)
    arg_ignore_size(nude_parser)

    arg_color_transfer(nude_parser)
    arg_compress(nude_parser)
    arg_image_size(nude_parser)

    arg_preferences(nude_parser)

    arg_cpu(processing_mod)
    arg_gpu(processing_mod)
    arg_checkpoints(nude_parser)
    arg_n_core(nude_parser)

    arg_json_args(nude_parser)
    arg_json_folder_name(nude_parser)

    arg_help(nude_parser)
    arg_debug(nude_parser)
    arg_version(nude_parser)

    return nude_parser


def set_args_nude_parser(args):
    set_arg_checkpoints(args)
    set_arg_preference(args)
    set_gpu_ids(args)


def check_args_nude_parser(parser, args):
    check_arg_input(parser, args)
    check_arg_output(parser, args)
    check_arg_checkpoints(parser, args)


def arg_input(parser):
    parser.add_argument(
        "-i",
        "--input",
        help="Path directory to watching.",
        required=True
    )


def arg_output(parser):
    parser.add_argument(
        "-o",
        "--output",
        help="Path of directory where the transformed photo(s) will be saved.",
        required=True
    )


def check_arg_input(parser, args):
    if not args.input:
        parser.error("-i, --input INPUT is required.")
    if not os.path.isdir(args.input):
        parser.error("Input {} directory doesn't exist.".format(args.input))


def check_arg_output(parser, args):
    if not args.output:
        parser.error("-o, --output OUTPUT is required.")
    if not os.path.isdir(args.output):
        parser.error("Output {} directory doesn't exist.".format(args.output))
