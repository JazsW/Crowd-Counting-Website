import sqlite3 as sql

conn = sql.connect('database.db')

with sql.connect("database.db") as con:
	cur = con.cursor()
	cur.execute("DELETE FROM camera;")
	print ("Sucessfull")
	con.commit()
