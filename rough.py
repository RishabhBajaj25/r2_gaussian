import pickle

# Open the pickle file in read-binary mode ('rb')
with open("/media/rishabh/SSD_1/Data/send_data/180_max007_projs_old.pickle", "rb") as file:
    data = pickle.load(file)

# Print or inspect the data
print(data)
