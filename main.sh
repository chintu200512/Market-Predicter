#!/bin/bash
# Fully Automated Python Web Server & Cloudflare Tunnel Launcher

APP_FILE="app.py"
PORT="5000"

# Defaults (Automated)
MODE_CHOICE="1"  # 1 = custom Python script, 2 = http.server
ENABLE_TUNNEL="Y"

trap 'printf "\n[!] Stopping services...\n"; stop' 2

banner() {
    clear
    printf "\e[1;94m"
    printf "=========================================\n"
    printf "  Automated App Tunnel Launcher          \n"
    printf "=========================================\n"
    printf "\e[0m\n"
}

check_dependencies() {
    command -v python3 > /dev/null 2>&1 || { echo >&2 "[!] Python 3 is required. Aborting."; exit 1; }
    command -v wget > /dev/null 2>&1 || { echo >&2 "[!] wget is required. Aborting."; exit 1; }
}

stop() {
    pkill -f "cloudflared" > /dev/null 2>&1
    pkill -f "$APP_FILE" > /dev/null 2>&1
    pkill -f "http.server $PORT" > /dev/null 2>&1
    printf "\e[1;32m[+] All processes stopped.\e[0m\n"
    exit 0
}

setup_cloudflared() {
    if [[ ! -f "./cloudflared" ]]; then
        printf "\e[1;93m[*] Cloudflared not found. Downloading automatically...\e[0m\n"
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
        printf "\e[1;32m[+] Cloudflared ready.\e[0m\n"
    fi
}

start_server() {
    printf "\e[1;92m[+] Launching local Python server on port %s...\e[0m\n" "$PORT"
    rm -f app.log

    if [[ "$MODE_CHOICE" == "1" ]]; then
        if [[ ! -f "$APP_FILE" ]]; then
            printf "\e[1;31m[!] Error: %s not found in current directory!\e[0m\n" "$APP_FILE"
            exit 1
        fi
        PORT="$PORT" python3 "$APP_FILE" > app.log 2>&1 &
    else
        python3 -m http.server "$PORT" > app.log 2>&1 &
    fi

    sleep 2
}

start_tunnel() {
    if [[ "$ENABLE_TUNNEL" =~ ^[Yy]$ ]]; then
        setup_cloudflared
        printf "\e[1;92m[+] Starting Cloudflare Tunnel...\e[0m\n"
        rm -f cf.log
        ./cloudflared tunnel --url "http://127.0.0.1:$PORT" --logfile cf.log > /dev/null 2>&1 &
        
        printf "\e[1;90m[*] Waiting for public URL...\e[0m\n"
        
        link=""
        for i in {1..12}; do
            sleep 1
            if [[ -f "cf.log" ]]; then
                link=$(grep -o 'https://[-0-9a-z]*\.trycloudflare.com' "cf.log" | head -n1)
                [[ -n "$link" ]] && break
            fi
        done
        
        if [[ -z "$link" ]]; then
            printf "\e[1;31m[!] Failed to retrieve public URL. Inspect 'cf.log'.\e[0m\n"
        else
            printf "\n\e[1;32m=============================================\e[0m\n"
            printf " \e[1;93mPublic Link:\e[0m \e[1;77m%s\e[0m\n" "$link"
            printf "\e[1;32m=============================================\e[0m\n"
        fi
    else
        printf "\e[1;32m[+] Server active at: http://127.0.0.1:%s\e[0m\n" "$PORT"
    fi

    printf "\n\e[1;90m[*] Server running in background. Stream logs with: tail -f app.log\e[0m\n"
    while true; do
        sleep 1
    done
}

# Main Execution Flow
banner
check_dependencies
start_server
start_tunnel