The file nvector_serial_custom_malloc.c contains an NVectorSerial implementation
patched to call a custom malloc function to allocate and free memory for the data arrays.

This file has been produces based on the original implementation stored in nvector_serial.c_orig.
Should the original implementation change, the changes can be merged into nvector_serial_custom_malloc.c
using a 3-way diff tool such as kdiff3.

UPDATE 28 Nov 2012:

In the 2.5 release of SUNDIALS, the file nvector_serial.c is unchanged compared to 2.4. No changes to
nvector_serial_custom_malloc.c needed for 2.5 compatibility.
