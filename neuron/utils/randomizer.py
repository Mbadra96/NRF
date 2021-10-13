import random
from typing import List, Any, overload

class Randomizer:

    @staticmethod
    def seed(seed):
        random.seed(seed)
    

    @staticmethod
    @overload
    def Float() -> float:
        ...

    @staticmethod
    @overload
    def Float(min:float,max:float) -> float:
        ...

    @staticmethod
    def Float(*args,**kwargs):
        if len(args) == 0 :
            return random.random()
        if args[0] == args[1]:
            return args[0]
        return random.random()*(args[1] - args[0]) + args[0]

    @staticmethod
    @overload
    def Integer()-> int :
        ...

    @staticmethod
    @overload
    def Integer(min:int,max:int) -> int:
        ...

    @staticmethod
    def Integer(*args,**kwarg) -> int:
        if len(args) == 0 :
            return random.randint(0,9)
        return random.randint(args[0], args[1])

    @staticmethod   
    def choice(values:List[Any]):
        if len(values) == 0:
            return None
        return random.choice(values)
    

if __name__ == '__main__':
    # Randomizer.seed(0)
    print(Randomizer.Integer(3,400))
