
from p2g.tasks.radqa import RadQA
from p2g.tasks.mimicqa import MIMICQA

def get_task(args):
    if args.task.startswith('radqa'):
        return RadQA(args)
    if args.task.startswith('mimicqa'):
        return MIMICQA(args)
    else:
        raise NotImplementedError(f'Task "{args.task}" not implemented')