"""python -m vlens data WORK N_COCO N_CC3M | train WORK RUN STEPS MINUTES | cycle WORK RUN STEPS MINUTES | eval WORK RUN
                | explore ADAPTER OUT [N_PHOTOS] | robust RUN [N] | native OUT [N_PHOTOS]
                | qimg OUT [N_TRAIN] | depth OUT [N_FIT] | pnla OUT MAPS [N_TRAIN] | swap OUT"""
import sys


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    if not a or a[0] not in ('data', 'train', 'cycle', 'eval', 'explore', 'robust', 'native', 'qimg', 'depth', 'pnla', 'swap'):
        sys.exit(__doc__)
    if a[0] == 'data':
        from .data import data
        data(a[1], int(a[2]), int(a[3]))
    elif a[0] in ('train', 'cycle'):
        from .train import train
        train(a[1], a[2], int(a[3]), float(a[4]), cycle=a[0] == 'cycle')
    elif a[0] == 'native':
        from .native import native
        native(a[1], *(int(x) for x in a[2:3]))
    elif a[0] == 'qimg':
        from .qimg import qimg
        qimg(a[1], *(int(x) for x in a[2:3]))
    elif a[0] == 'depth':
        from .depth import depth
        depth(a[1], *(int(x) for x in a[2:3]))
    elif a[0] == 'swap':
        from .swap import swap
        swap(a[1])
    elif a[0] == 'pnla':
        from .pnla import pnla
        pnla(a[1], a[2], *(int(x) for x in a[3:4]))
    elif a[0] == 'robust':
        from .robust import robust
        robust(a[1], *(int(x) for x in a[2:3]))
    elif a[0] == 'explore':
        from .explore import explore
        explore(a[1], a[2], *(int(x) for x in a[3:4]))
    else:
        from .evaluate import evaluate
        evaluate(a[1], a[2])


if __name__ == '__main__':
    main()
