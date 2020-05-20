import pyodbc


class DataAccess:

    def __init__(self, cs):
        self.cnxn = pyodbc.connect(cs)
        self.cursor = self.cnxn.cursor()

    def update_result(self, throw_id, result):
        self.cursor.execute(
            "UPDATE Throws " +
            "SET Result = ?, ResultProcessedDate = GETUTCDATE() " +
            "WHERE Id = ?", result, throw_id)
        self.cnxn.commit()

    def is_throw_exists(self, throw_id):
        self.cursor.execute(
          "SELECT 1 FROM Throws WHERE Id = ?", throw_id)
        return self.cursor.fetchone() is not None

    def is_experiment_exists(self, experiment_id):
        self.cursor.execute(
          "SELECT 1 FROM Experiments WHERE Id = ?", experiment_id)
        return self.cursor.fetchone() is not None

    def insert_experiment(self, experiment_id):
        self.cursor.execute(
            "INSERT INTO Experiments(Id, Name, NumberOfDices) VALUES (?,?,?)",
            experiment_id, 'Six Dices', 6)
        self.cnxn.commit()

    def insert_throw(self, throw_id):
        self.cursor.execute("INSERT INTO Throws (Id) VALUES (?)", throw_id)
        self.cnxn.commit()

    def insert_prediction(self, di, throw_id, prediction, confidence):
        throw_ids = [throw_id for my_object in prediction]
        zipped = zip(di, throw_ids, prediction, confidence)
        zipped = list(zipped)
        self.cursor.executemany(
            "INSERT INTO " +
            "Predictions (Id, ThrowId, Prediction, Confidence) " +
            "VALUES (?,?,?,?)", zipped)
        self.cnxn.commit()
