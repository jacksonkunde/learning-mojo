import random

@fieldwise_init
struct Grid(Copyable, Movable, Stringable):
    var rows: Int
    var cols: Int
    var grid: List[List[Int]]

    fn evolve(self) -> Self:
        next_generation = List[List[Int]]()

        for row in range(self.rows):
            row_data = List[Int]()

            row_above = (row - 1) % self.rows
            row_below = (row + 1) % self.rows

            for col in range(self.cols):
                col_left = (col - 1) % self.cols
                col_right = (col + 1) % self.cols

                num_neighbors = (
                    self[row_above, col_left]
                    + self[row_above, col]
                    + self[row_above, col_right]
                    + self[row, col_left]
                    + self[row, col_right]
                    + self[row_below, col_left]
                    + self[row_below, col]
                    + self[row_below, col_right]
                )

                new_state = 0
                if self[row, col] == 1 and (
                    num_neighbors == 2 or num_neighbors == 3
                ):
                    new_state = 1
                elif self[row, col] == 0 and num_neighbors == 3:
                    new_state = 1
                row_data.append(new_state)

            next_generation.append(row_data^)

        return Self(self.rows, self.cols, next_generation^)

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


    @staticmethod
    fn random(rows: Int, cols: Int, p_alive: Float64 = 0.5) -> Self:
        # Set random seed based on current time
        random.seed()

        var grid: List[List[Int]] = []

        for _ in range(rows):
            var row_data: List[Int] = []
            for _ in range(cols):
                var cell_val: Int = 1 if random.random_float64(0, 1) < p_alive else 0
                row_data.append(cell_val)
            grid.append(row_data^)
            
        return Self(rows, cols, grid^)
