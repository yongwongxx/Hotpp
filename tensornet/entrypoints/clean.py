import os


def main(*args, **kwargs):
    i = 0
    while os.path.exists("results{}".format(i)):
        i += 1
    os.mkdir("results{}".format(i))
    os.system("mv allpara.yaml log.txt *err* *out* *.pt results{}".format(i))
    os.system("cp input.yaml *.pt results{}".format(i))
    print("Done!")


if __name__ == "__main__":
    main()

