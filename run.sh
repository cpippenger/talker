
main() {
    run_jupyter
}

run_jupyter() {
	#mkdir -p /notebooks
	#cd /notebooks
	jupyter notebook --ip=0.0.0.0 --allow-root --no-browser &
}


log_i() {
    log "[INFO] ${@}"
}

log_w() {
    log "[WARN] ${@}"
}

log_e() {
    log "[ERROR] ${@}"
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${@}"
}

control_c() {
    echo ""
    exit
}

trap control_c SIGINT SIGTERM SIGHUP

main

exit
