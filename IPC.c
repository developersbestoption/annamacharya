#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <sys/wait.h>

#define MSG_SIZE 100

void signal_handler(int sig) {
    printf("Child received signal: SIGUSR1\n");
}

int main() {
    int fd[2];
    pid_t pid;
    char buffer[MSG_SIZE];

    if (pipe(fd) == -1) {
        perror("pipe");
        exit(EXIT_FAILURE);
    }

    pid = fork();

    if (pid < 0) {
        perror("fork");
        exit(EXIT_FAILURE);
    }

    if (pid == 0) {
        // Child Process
        signal(SIGUSR1, signal_handler);
        close(fd[0]); // Close read end

        const char *message = "Hello from child";
        write(fd[1], message, strlen(message) + 1);
        close(fd[1]);

        printf("Child process started. PID: %d\n", getpid());
        pause();  // Wait for signal from parent
        exit(0);
    } else {
        // Parent Process
        close(fd[1]); // Close write end

        printf("Parent: Reading from pipe...\n");
        read(fd[0], buffer, sizeof(buffer));
        close(fd[0]);

        printf("Parent received: %s\n", buffer);
        printf("Parent: Sending SIGUSR1 to child...\n");
        kill(pid, SIGUSR1);

        wait(NULL);  // Wait for child to finish
    }

    return 0;
}
