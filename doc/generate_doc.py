import os

module_import_error = """
Unable to import the %s module
Error: %s
Did you forget to update your PYTHONPATH variable?"""

modules = open("modules.txt", "r").readlines()
for module in modules:
    module = module.split()
    name = ' '.join(module[:-1])
    filename = '_'.join(module[:-1]) + '.rst'
    module_name = module[-1]

    try:
        exec("import %s" % module_name)
    except Exception as what:
        raise ImportError(module_import_error % (module_name, what))

    if not os.path.isdir("modules"):
        os.mkdir("modules")
    
    print "Writing modules/%s" % filename

    outfile = open("modules/%s" % filename, "w")
    outfile.write(name + "\n")
    for characters in name:
        outfile.write("=")
    outfile.write("\n\n")
    outfile.write(".. automodule:: %s\n" % module_name)
    outfile.write("\t:members:\n")
    outfile.write("\t:undoc-members:\n")
