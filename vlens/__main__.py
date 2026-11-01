"""python -m vlens data WORK N_COCO N_CC3M | train WORK RUN STEPS MINUTES | cycle WORK RUN STEPS MINUTES | eval WORK RUN"""
import sys


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    if not a or a[0] not in ('data', 'train', 'cycle', 'eval'):
        sys.exit(__doc__)
    if a[0] == 'data':
        from .data import data
        data(a[1], int(a[2]), int(a[3]))
    elif a[0] in ('train', 'cycle'):
        from .train import train
        train(a[1], a[2], int(a[3]), float(a[4]), cycle=a[0] == 'cycle')
    else:
        from .evaluate import evaluate
        evaluate(a[1], a[2])


if __name__ == '__main__':
    main()
