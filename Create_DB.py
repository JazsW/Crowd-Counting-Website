import sqlite3 as sql

conn = sql.connect('database.db')
print ("Opened database successfully")

conn.execute("CREATE TABLE location (locationID INTEGER PRIMARY KEY, location VARCHAR (30) NOT NULL, max_occupancy INTEGER NOT NULL);")
print ("Location table created successfully")

conn.execute("CREATE TABLE camera (cameraID INTEGER PRIMARY KEY, locationID INTEGER, count INTEGER, FOREIGN KEY (locationID) REFERENCES location (locationID) );")
print ("Camera table created successfully")

conn.close()

with sql.connect("database.db") as con:
	cur = con.cursor()
	cur.execute("INSERT INTO location (location, max_occupancy) VALUES ('Tent 1', 300)")
	print ("Content added successfully")
	cur.execute("INSERT INTO location (location, max_occupancy) VALUES ('Tent 2', 100)")
	print ("Content added successfully")
	cur.execute("INSERT INTO location (location, max_occupancy) VALUES ('Tent 3', 100)")
	print ("Content added successfully")
	
	con.commit()
	print("Record successfully added")
