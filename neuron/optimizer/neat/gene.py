from multipledispatch import dispatch
class Gene:
    @dispatch(int)
    def __init__(self,innovation_number) -> None:
        self.innovation_number = innovation_number
    
    @dispatch(object)
    def __init__(self,other:'Gene') -> None:
        if isinstance(other, Gene):
            self.innovation_number = other.innovation_number
        else:
            raise TypeError(f"invaled object excepted Gene Type but {other.__class__.__name__} was given instead")

    def __eq__(self, other: 'Gene') -> bool:
        """Overrides the default implementation"""
        if isinstance(other, Gene):
            return self.innovation_number == other.innovation_number
        return False

    def clone(self) -> 'Gene':
        return Gene(self.innovation_number)
    
    def __str__(self) -> str:
        return f"Gene {{innovationNumber = {self.innovation_number}}}"
