from is_even_nn import *

if __name__ == "__main__":
    TYPE = "int"

    if TYPE == "float":
        _, tdata, tlabels = generate_floats(100000)
        _, edata, elabels = generate_floats(100)
    if TYPE == "int":
        _, tdata, tlabels = generate_ints(100000)
        _, edata, elabels = generate_ints(100)
    else:
        raise NotImplementedError("Unknown type " + TYPE)

    net = IsEvenNN()
    net.train(tdata, tlabels, 10)
    torch.save(net.net.state_dict(), "isEvenModel.pt")
    net.predict(edata)
