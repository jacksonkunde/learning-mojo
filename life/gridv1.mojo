@fieldwise_init
struct Grid(Copyable, Movable):
    var rows: Int
    var cols: Int
    var grid: List[List[Int]]

    fn grid_str(self) -> String:
        # Define an empty string
        ret_str = String()
        
        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row][col] == 1:
                    ret_str += "*"
                else:
                    ret_str += " "
            if row < self.rows:
                ret_str += "\n"
        
        return ret_str
