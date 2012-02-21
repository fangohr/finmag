The file nvector_serial_custom_malloc.c contains an NVectorSerial implementation
patched to call a custom malloc function to allocate and free memory for the data arrays.

This file has been produces based on the original implementation stored in nvector_serial.c_orig.
Should the original implementation change, the changes can be merged into nvector_serial_custom_malloc.c
using a 3-way diff tool such as kdiff3.
