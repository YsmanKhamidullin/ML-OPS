import requests
import csv
import os
import sys

currentDirectory = os.path.dirname(os.path.abspath(sys.argv[0]))
dsName = "/dataset.csv"
def writeDataToDisk(path,data):
    with open(path, "wb") as f:
        f.write(data)


def writeCsvToDisk(path, csvData):
    with open(path, "w") as csvFile:
        csvW = csv.writer(csvFile)
        csvW.writerows(csvData)

def downloadData():
    url = "https://gitlab.com/datascience-book/code/-/raw/master/data/titanic.csv?inline=false"
    r = requests.get(url)
    writeDataToDisk(currentDirectory+dsName, r.content)
    
downloadData()

dataRows = []
with open(currentDirectory+dsName) as csvFile:
    readedCsv = csv.reader(csvFile)
    
    for row in readedCsv:
        dataRows.append(row)
        
    testData = dataRows[:len(dataRows)//2]
    writeCsvToDisk(currentDirectory + "/test.csv", testData)
    
    trainData = dataRows[len(dataRows)//2:]
    writeCsvToDisk(currentDirectory + "/train.csv", trainData)