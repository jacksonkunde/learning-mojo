@fieldwise_init
struct Grid(Copyable, Movable, Stringable):
    var rows: Int
    var cols: Int
    var grid: List[List[Int]]

    fn __getitem__(self, row: Int, col: Int) -> Int:
        return self.grid[row][col]

    fn __setitem__(mut self, row: Int, col: Int, value: Int):
        self.grid[row][col] = value

    fn __str__(self) -> String:
        # Define an empty string
        ret_str = String()
        
        for row in range(self.rows):
            for col in range(self.cols):
                if self[row, col] == 1:
                    ret_str += "*"
                else:
                    ret_str += " "
            if row < self.rows:
                ret_str += "\n"
        
        return ret_str
