import torch
import argparse
# from polyaxon_client import tracking

def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts backbone weights from a checkpoint')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        'output', type=str, help='destination file name')
    args = parser.parse_args()
    return args


def main():
    # base_path = tracking.get_data_paths()['ceph'] + '/'
    # output_dir = tracking.get_outputs_path()

    args = parse_args()
    # args.checkpoint = base_path + args.checkpoint
    # args.output = output_dir + args.output
    assert args.output.endswith(".pth")
    ck = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    output_dict = dict(state_dict=dict(), author="OpenSelfSup")
    has_backbone = False
    for key, value in ck['state_dict'].items():
        if key.startswith('backbone'):
            output_dict['state_dict'][key[9:]] = value
            has_backbone = True
    if not has_backbone:
        raise Exception("Cannot find a backbone module in the checkpoint.")
    torch.save(output_dict, args.output, _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    main()
