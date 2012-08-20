/* from googling for how to read Mac address

Seems to work, but needs to be told which card we care about (eth0, eth1, ...)

Could always sweep all network adapter names we can think of, and if there is any match with the registered Mac address, continue?

*/

#include <sys/socket.h>
#include <sys/ioctl.h>
#include <linux/if.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(void)
{
  struct ifreq s;
  int fd = socket(PF_INET, SOCK_DGRAM, IPPROTO_IP);

  strcpy(s.ifr_name, "eth0");
  if (0 == ioctl(fd, SIOCGIFHWADDR, &s)) {
    int i;
    for (i = 0; i < 6; ++i)
      printf(" %02x", (unsigned char) s.ifr_addr.sa_data[i]);
    puts("\n");
    return EXIT_SUCCESS;
  }
  return EXIT_FAILURE;
}

