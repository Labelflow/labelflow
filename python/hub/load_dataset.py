import hub
from PIL import Image


def main():
    ds = hub.load("s3://labelflow-hub-1/test-yolo-2021-12-22T111211", read_only=True)
    # Indexing
    W = ds.images[0].numpy()  # Fetch an image and return a NumPy array
    X = ds.labels[0].numpy(aslist=True)  # Fetch a label and store it as a
    print("label = \n", X)
    # list of NumPy arrays
    im = Image.fromarray(W)
    im.save("your_file.jpeg")
    return


main()
