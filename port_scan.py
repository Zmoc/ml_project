import re
import socket
import subprocess

import psutil


def run_nmap_scan():
    print("[*] Running nmap TCP scan on localhost...")
    result = subprocess.run(
        ["nmap", "-sS", "-p-", "127.0.0.1"], capture_output=True, text=True, shell=True
    )
    return result.stdout


def parse_nmap_output(nmap_output):
    open_ports = []
    for line in nmap_output.splitlines():
        if re.match(r"^\d+/tcp\s+open", line):
            parts = line.split()
            port_proto = parts[0]
            service = parts[2] if len(parts) > 2 else "unknown"
            open_ports.append((port_proto, service))
    return open_ports


def get_processes_by_port():
    port_process_map = {}

    for conn in psutil.net_connections(kind="inet"):
        try:
            port = conn.laddr.port
            proto = "tcp" if conn.type == socket.SOCK_STREAM else "udp"
            key = f"{port}/{proto}"

            pid = conn.pid
            pname = psutil.Process(pid).name() if pid else "Unknown"

            port_process_map[key] = {
                "pid": pid,
                "process": pname,
                "status": conn.status,
            }

        except Exception as e:
            continue

    return port_process_map


def generate_report(open_ports, process_map):
    report_lines = []
    for port_proto, service in open_ports:
        process_info = process_map.get(
            port_proto, {"pid": None, "process": "Unknown", "status": "Not found"}
        )
        report_lines.append(
            f"{port_proto:<10} | {service:<15} | PID: {str(process_info['pid']):<6} "
            f"| Process: {process_info['process']:<20} | Status: {process_info['status']}"
        )
    return "\n".join(report_lines)


def main():
    nmap_output = run_nmap_scan()
    open_ports = parse_nmap_output(nmap_output)
    process_map = get_processes_by_port()
    report = generate_report(open_ports, process_map)

    print("\n🧾 Port Scan Report:\n")
    print("PORT       | SERVICE         | PID     | PROCESS              | STATUS")
    print("-" * 75)
    print(report)


if __name__ == "__main__":
    main()
