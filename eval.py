from is_even_nn import *

if __name__ == "__main__":
    TYPE = "int"

    if TYPE == "float":
        eraw, edata, elabels = generate_floats(100)
    if TYPE == "int":
        eraw, edata, elabels = generate_ints(100)
    else:
        raise NotImplementedError("Unknown type " + TYPE)

    net = IsEvenNN()
    net.net.load_state_dict(torch.load("isEvenModel.pt"))
    predictions = net.predict(edata)
    for num, is_even in zip(eraw, predictions):
        print(f"{num} is {'even' if is_even else 'not even'}")

    print(f"\nAccuracy: {net.accuracy(edata,elabels)} ({len(edata)} samples)")
