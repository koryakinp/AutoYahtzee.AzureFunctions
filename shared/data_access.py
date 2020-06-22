import pyodbc
import uuid


class DataAccess:

    def __init__(self, cs):
        self.cnxn = pyodbc.connect(cs)
        self.cursor = self.cnxn.cursor()

    def get_throw(self, throw_id):
        self.cursor.execute(
          "SELECT Id FROM Throws WHERE Id = ?", throw_id)
        rows = self.cursor.fetchall()

        return [row.Id for row in rows]

    def get_predictions(self, throw_id):
        self.cursor.execute(
          "SELECT Id FROM Predictions WHERE ThrowId = ?", throw_id)
        rows = self.cursor.fetchall()

        return [row.Id for row in rows]

    def insert_throw(self, throw_id):
        self.cursor.execute("INSERT INTO Throws (Id) VALUES (?)", throw_id)
        self.cnxn.commit()

    def delete_throw(self, throw_id):
        self.cursor.execute("DELETE FROM Throws WHERE Id = ?", throw_id)
        self.cnxn.commit()

    def insert_prediction(self, throw_id, predictions, confidences, prediction_ids):
        throw_ids = [throw_id for _ in predictions]
        zipped = zip(prediction_ids, throw_ids, predictions, confidences)
        zipped = list(zipped)
        self.cursor.executemany(
            "INSERT INTO " +
            "Predictions (Id, ThrowId, Prediction, Confidence) " +
            "VALUES (?,?,?,?)", zipped)
        self.cnxn.commit()
