#!/bin/bash
# Python Web Server & Tunnel Launcher

# Gracefully handle Ctrl+C (SIGINT)
trap 'printf "\n[!] Stopping services...\n"; stop' 2

APP_FILE="app.py"     # <--- Updated to match your file
PORT="5000"           # <--- Updated to match Flask

banner() {
    clear
    printf "\e[1;94m"
    printf "=========================================\n"
    printf "      Python App Tunnel Launcher         \n"
    printf "=========================================\n"
    printf "\e[0m\n"
}

check_dependencies() {
    command -v python3 > /dev/null 2>&1 || { echo >&2 "[!] Python 3 is required but not installed. Aborting."; exit 1; }
    command -v wget > /dev/null 2>&1 || { echo >&2 "[!] wget is required but not installed. Aborting."; exit 1; }
}

stop() {
    # Terminate background Cloudflared and Python processes
    pkill -f "cloudflared" > /dev/null 2>&1
    pkill -f "$APP_FILE" > /dev/null 2>&1
    pkill -f "http.server $PORT" > /dev/null 2>&1
    printf "\e[1;32m[+] All processes stopped.\e[0m\n"
    exit 0
}

setup_cloudflared() {
    if [[ ! -f "./cloudflared" ]]; then
        printf "\e[1;93m[*] Cloudflared binary not found. Downloading...\e[0m\n"
        arch=$(uname -m)
        arch2=$(uname -a | grep -o 'Android' | head -n1)

        if [[ $arch == *'arm'* ]] || [[ $arch2 == *'Android'* ]]; then
            wget -q --no-check-certificate https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm -O cloudflared
        elif [[ "$arch" == *'aarch64'* ]]; then
            wget -q --no-check-certificate https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64 -O cloudflared
        elif [[ "$arch" == *'x86_64'* ]]; then
            wget -q --no-check-certificate https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
        else
            wget -q --no-check-certificate https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-386 -O cloudflared
        fi
        chmod +x cloudflared
        printf "\e[1;32m[+] Cloudflared downloaded successfully.\e[0m\n"
    fi
}

start_server() {
    printf "\n\e[1;93mSelect execution mode:\e[0m\n"
    printf " [1] Run custom Python script (\e[1;36m%s\e[0m)\n" "$APP_FILE"
    printf " [2] Serve HTML files in current folder via Python HTTP module\n"
    read -p "Option [1/2] (Default: 1): " mode_choice
    mode_choice="${mode_choice:-1}"

    printf "\e[1;92m[+] Starting local Python server on port %s...\e[0m\n" "$PORT"

    if [[ "$mode_choice" == "1" ]]; then
        if [[ ! -f "$APP_FILE" ]]; then
            printf "\e[1;31m[!] Error: %s not found in current directory!\e[0m\n" "$APP_FILE"
            exit 1
        fi
        # Passes the port securely as an environment variable
        PORT="$PORT" python3 "$APP_FILE" > /dev/null 2>&1 &
    else
        # Built-in static file server for serving static HTML/JS/CSS
        python3 -m http.server "$PORT" > /dev/null 2>&1 &
    fi

    sleep 2
}

start_tunnel() {
    read -p $'\n\e[1;93mExpose server to public internet via Cloudflare Tunnel? [Y/n]: \e[0m' tunnel_opt
    tunnel_opt="${tunnel_opt:-Y}"

    if [[ "$tunnel_opt" =~ ^[Yy]$ ]]; then
        setup_cloudflared
        printf "\e[1;92m[+] Starting Cloudflare Tunnel...\e[0m\n"
        rm -f cf.log
        ./cloudflared tunnel --url "http://127.0.0.1:$PORT" --logfile cf.log > /dev/null 2>&1 &
        
        sleep 6
        
        link=$(grep -o 'https://[-0-9a-z]*\.trycloudflare.com' "cf.log" | head -n1)
        
        if [[ -z "$link" ]]; then
            printf "\e[1;31m[!] Could not generate public URL. Check cf.log for details.\e[0m\n"
        else
            printf "\n\e[1;32m=============================================\e[0m\n"
            printf " \e[1;93mPublic Link:\e[0m \e[1;77m%s\e[0m\n" "$link"
            printf "\e[1;32m=============================================\e[0m\n"
        fi
    else
        printf "\e[1;32m[+] Server active locally at: http://127.0.0.1:%s\e[0m\n" "$PORT"
    fi

    printf "\n\e[1;90m[*] Server running. Press Ctrl + C to stop.\e[0m\n"
    while true; do
        sleep 1
    done
}

# Execution Flow
banner
check_dependencies
start_server
start_tunnel