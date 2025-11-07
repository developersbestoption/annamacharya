#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CMD_LEN 1024

int main() {
    char command[MAX_CMD_LEN];

    while (1) {
        printf("> ");  // Display prompt
        if (fgets(command, sizeof(command), stdin) == NULL) {
            perror("fgets failed");
            continue;
        }

        // Remove newline character
        size_t len = strlen(command);
        if (len > 0 && command[len - 1] == '\n') {
            command[len - 1] = '\0';
        }

        // Exit condition
        if (strcmp(command, "exit") == 0) {
            printf("Exiting Windows shell...\n");
            break;
        }

        // Execute command using Windows system shell
        int status = system(command);
        if (status == -1) {
            printf("Command execution failed.\n");
        }
    }

    return 0;
}
