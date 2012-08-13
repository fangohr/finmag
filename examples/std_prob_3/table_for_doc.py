import os
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def write_table():
    with open(os.path.join(MODULE_DIR, "data_energies.txt")) as f:
        lines = f.readlines()
    vor = lines[-2].split()
    flo = lines[-1].split()

    with open(os.path.join(MODULE_DIR, "table_template.txt")) as f:
        table_template = f.read()

    with open(os.path.join(MODULE_DIR, "doc_table.rst"), "w") as f:
        f.write(table_template.format(
            float(vor[5]), float(vor[3]), float(vor[4]), float(vor[2]),
            float(flo[5]), float(flo[3]), float(flo[4]), float(flo[2])))
            
if __name__ == "__main__":
    write_table()
