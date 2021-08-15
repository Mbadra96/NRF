import random
from multipledispatch import dispatch

class Randomizer:
    seed:'int' = 0


    @staticmethod
    def seed(seed):
        random.seed(seed)
    
    @staticmethod
    @dispatch()
    def Float() -> float:
        return random.random()

    @staticmethod  
    @dispatch(float, float)
    def Float(min,max) -> float:
        return random.random()*(max - min) + min
    
    @staticmethod
    @dispatch()
    def Integer()-> int :
        return random.randint(0,9)

    @staticmethod
    @dispatch(int,int)
    def Integer(min,max) -> int:
        return random.randint(min, max)
    
    def choice(values):
        if len(values) == 0:
            return None
        return random.choice(values)
    

if __name__ == '__main__':
    # Randomizer.seed(0)
    print(Randomizer.Integer(3,400))
