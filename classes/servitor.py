import contextlib
import os
import sys
import nmap
from scapy.all import IP, TCP, wrpcap
import random
import re


class Brain:
    def __init__(self, model_path):
        self.brain_path = model_path


class Servitor:
    def __init__(self, config_path, brain: Brain = None):
        self.config_path = config_path
        self.brain = brain

        if brain:
            self.brain_exist = True
            self.llm_startup(self.config_path, self.brain.brain_path)
            self.get_input()
        else:
            self.brain_exist = False
            print("No brain detected. Quitting...")

    def llm_startup(self, config_path, brain_path):
        import llama_cpp
        import yaml
        from llama_cpp import Llama

        self.persona = {
            "role": "system",
            "content": "You are a cybersecurity agent tasked with defending the local system. You are knowledgable and patient.",
        }
        llama_cpp.llama_print_system_info = lambda: b""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        model_path = brain_path
        n_threads = config["llm"]["n_threads"]
        device = config["llm"]["device"]
        print("Please wait...loading brain...")
        with suppress_stdout_stderr():
            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=n_threads,
                n_gpu_layers=0,
            )
        print(
            f"Configurations loaded, Model: {model_path}, Threads: {n_threads}, Device: {device}"
        )
        print("Brain loaded")
        print(
            "My name is Servitor. I am a cybersecurity agent tasked with defending your local system. "
            "I can currently scan your local system for open ports, "
            "simulate a DDoS attack, "
            "scan network traffic, "
            "and summarize saved logs."
        )

    def get_input(self):
        print("Servitor is ready. Type your command:")

        actions = {
            "scan_ports": self.scan_ports,
            "simulate_ddos": self.sim_ddos,
            "scan_traffic": self.scan_traffic,
            "summarize_logs": self.summarize_logs,
            "ignore": self.handle_ignore,
            "request_clarification": self.handle_request_clarification,
        }

        while True:
            try:
                user_input = input(">:")
            except (KeyboardInterrupt, EOFError):
                print("\nAgent exiting.")
                break

            if user_input.lower() in ["exit", "quit"]:
                print("Servitor quitting...")
                break

            intent = self.intent(user_input)

            try:
                action = actions.get(intent, self.handle_unrecognized)
                action()
            except Exception as e:
                print(f"Error: {e}")

    def scan_ports(self):
        scan_result, scan_prompt = self.port_scan()
        try:
            print("Generating report summary...")
            summary = self.summarize_report(scan_result, scan_prompt)
            print("\nSummary:\n", summary)
        except Exception as e:
            print(f"[!] Failed to summarize report: {e}")

    def summarize_logs(self):
        with open("data/logs/logs.txt", "r") as f:
            text = f.readlines()

        log_text = {"role": "user", "content": text}
        log_prompt = {
            "role": "user",
            "content": "This file includes a variety of logs such as error messages, warnings, and informational logs that were seen in the system.",
        }
        try:
            print("Generating report summary...")
            summary = self.summarize_report(log_text, log_prompt)
            print("\nSummary:\n", summary)
        except Exception as e:
            print(f"[!] Failed to summarize report: {e}")

    def sim_ddos(self):
        self.sim_syn_flood(
            target_ip="127.0.0.1",
            target_port="80",
            num_packets=1000,
            output_pcap="data/shark/log.pcap",
        )

    def scan_traffic(self):
        scan_result, scan_prompt = self.traffic_scan()
        try:
            print("Generating report summary...")
            summary = self.summarize_report(scan_result, scan_prompt)
            print("\nSummary:\n", summary)
        except Exception as e:
            print(f"Failed to summarize report: {e}")

    def handle_ignore(self):
        print("No action required.")

    def handle_request_clarification(self):
        print("I need more context to understand the command.")

    def handle_unrecognized(self):
        print("Unrecognized action. Please try again.")

    def port_scan(self):
        nm = nmap.PortScanner()
        target_ip = "127.0.0.1"
        print(f"Scanning all major ports on {target_ip}...")

        nm.scan(
            hosts=target_ip,
            arguments="-p 20,21,22,23,25,53,80,443",
        )

        scan_result = ""

        for host in nm.all_hosts():
            scan_result += f"Host {host} is {nm[host].state()}\n"
            for proto in nm[host].all_protocols():
                scan_result += f"Protocol: {proto}\n"
                lport = nm[host][proto].keys()
                for port in lport:
                    port_state = nm[host][proto][port]["state"]
                    service_name = nm[host][proto][port].get("name", "Unknown service")
                    service_version = nm[host][proto][port].get(
                        "version", "Unknown version"
                    )

                    scan_result += (
                        f"Port {port} is {port_state}, "
                        f"Service: {service_name}, Version: {service_version}\n"
                    )

        scan_dict = {"role": "user", "content": scan_result}

        report_type_prompt = {
            "role": "system",
            "content": "These are the results of an NMAP port scan. They describe the status of all ports on the system and what services are currently running.",
        }
        return scan_dict, report_type_prompt

    def sim_syn_flood(self, target_ip, target_port, num_packets, output_pcap):
        packets = []

        for _ in range(num_packets):
            fake_src_ip = ".".join(str(random.randint(1, 254)) for _ in range(4))
            ip = IP(src=fake_src_ip, dst=target_ip)
            syn = TCP(
                sport=random.randint(1024, 65535),
                dport=int(target_port),
                flags="S",
                seq=random.randint(1, 100000),
            )
            packet = ip / syn
            packets.append(packet)

        wrpcap(output_pcap, packets)
        print(
            f"Simulated {num_packets} SYN packets from random IPs to {target_ip}:{target_port} and saved to {output_pcap}"
        )

    def traffic_scan(self):
        traffic_results = self.evaluate_traffic("data/shark/log.pcap")
        report_type_prompt = {
            "role": "user",
            "content": "These are the results of a Wireshark pcap scan. They describe the number of packets that were captured and other information about them along with suspicion scores for SYN flood and a DDos attack.",
        }
        return traffic_results, report_type_prompt

    def evaluate_traffic(self, pcap_file):
        from collections import Counter
        import pyshark
        import nest_asyncio
        import time

        nest_asyncio.apply()

        cap = pyshark.FileCapture(pcap_file, display_filter="tcp")

        total_packets = 0
        syn_count = 0
        syn_packets = []
        source_ips = Counter()
        target_ips = Counter()
        ignored_packets = 0

        start_time = time.time()

        for packet in cap:
            try:
                total_packets += 1
                tcp_flags = int(packet.tcp.flags, 16)
                syn_flag_set = tcp_flags & 0x02
                ack_flag_set = tcp_flags & 0x10

                if syn_flag_set and not ack_flag_set:
                    src_ip = packet.ip.src
                    dst_ip = packet.ip.dst
                    syn_count += 1
                    syn_packets.append((src_ip, dst_ip))
                    source_ips[src_ip] += 1
                    target_ips[dst_ip] += 1

            except AttributeError:
                ignored_packets += 1
                continue

        cap.close()

        output = ""

        duration = time.time() - start_time
        output += f"Processed {total_packets} packets in {duration:.2f} seconds.\n"
        if ignored_packets > 0:
            output += f"Warning: Ignored {ignored_packets} packets due to missing IP/TCP fields.\n"

        output += "\n--- SYN Flood Evaluation ---\n"
        output += f"Total packets: {total_packets}\n"
        output += f"SYN packets without ACK: {syn_count}\n"

        syn_ratio = (syn_count / total_packets) * 100 if total_packets > 0 else 0
        output += f"SYN flood suspicion score: {syn_ratio:.2f}%\n"
        output += (
            "A SYN flood is a type of DoS attack that overwhelms a server by sending a "
            "large number of incomplete connection requests (SYN packets) with spoofed IP addresses, "
            "exhausting the server's resources and causing it to become unresponsive."
        )

        if syn_ratio > 50:
            output += "High chance of SYN flood detected.\n"
        else:
            output += "Traffic looks normal.\n"

        output += "\n--- DDoS Evaluation ---\n"
        output += f"Unique attacking IPs: {len(source_ips)}\n"
        output += f"Most targeted IP: {target_ips.most_common(1)}\n"
        output += (
            "A DDoS (Distributed Denial of Service) attack involves overwhelming a target system, "
            "network, or service with a massive volume of traffic from multiple compromised sources, "
            "causing it to become slow, unreliable, or completely unavailable."
        )
        if len(source_ips) > 10 and target_ips.most_common(1)[0][1] > 50:
            output += (
                "DDoS likely detected: multiple sources targeting one destination.\n"
            )
        else:
            output += "No strong evidence of DDoS attack.\n"

        output += "\n--- Additional Traffic Insights ---\n"
        output += f"Most active source IP: {source_ips.most_common(1)}\n"
        output += f"Most targeted destination IP: {target_ips.most_common(1)}\n"

        if len(source_ips) > 5 and len(target_ips) > 1:
            high_traffic_ips = [ip for ip, count in source_ips.items() if count > 50]
            if high_traffic_ips:
                output += (
                    "High-traffic source IPs detected: "
                    + ", ".join(high_traffic_ips)
                    + "\n"
                )

        traffic_report_dict = {"role": "user", "content": output}

        return traffic_report_dict

    def query(
        self, messages: list, temperature: float = 0.0, max_tokens: int = 512
    ) -> str:
        # response = self.llm(prompt,max_tokens=max_tokens,echo=False,stream=False,temperature=0.01,top_p=0.95,)
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            stream=False,
            temperature=temperature,
            top_p=0.95,
        )

        response_text = response["choices"][0]["message"]["content"].strip()
        response_text = "\n".join(
            [line.strip() for line in response_text.splitlines() if line.strip()]
        )
        if len(response_text) == 0:
            response_text = "No meaningful response generated."

        return response_text

    def intent(self, user_input: str, prefix: str = "", suffix: str = "") -> str:
        """Determines the user's intent based on input."""
        framed_input = f"{prefix.strip()} {user_input.strip()} {suffix.strip()}".strip()
        intent_prompt = [
            self.persona,
            {
                "role": "user",
                "content": "Decide what the following instruction is asking you to do.",
            },
            {"role": "user", "content": framed_input},
            {
                "role": "user",
                "content": "Respond with only one of the following words: scan_ports, simulate_ddos, scan_traffic, summarize_logs, ignore, request_clarification",
            },
            {"role": "user", "content": "Only use these keywords exactly."},
            {"role": "user", "content": "Do not explain your choice."},
            {"role": "user", "content": "Only respond with one word."},
        ]
        llm_response = self.query(intent_prompt)

        response_normalized = re.sub(r"\\_", "_", llm_response.strip().lower())

        print(f"Response used for intent: {response_normalized}")

        return {
            "scan_ports": "scan_ports",
            "simulate_ddos": "simulate_ddos",
            "scan_traffic": "scan_traffic",
            "summarize_logs": "summarize_logs",
            "ignore": "ignore",
            "request_clarification": "request_clarification",
        }.get(response_normalized, "unrecognized")

    def summarize_report(self, report_text: dict, report_prompt: dict) -> str:
        """Asks the LLM to summarize the port scan report."""

        summary_prompt = [
            self.persona,
            report_prompt,
            report_text,
            {
                "role": "user",
                "content": "Please summarize this report and highlight any items that might require a security review.",
            },
            {
                "role": "user",
                "content": "Explain it at to someone that doesn't have a strong background in cybersecurity.",
            },
            {
                "role": "user",
                "content": "Be specific and give actionable steps on how to address the issues.",
            },
            {
                "role": "user",
                "content": "When possible, give specific times and items to focus on.",
            },
        ]

        return self.query(summary_prompt, temperature=0.2, max_tokens=512)


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Suppresses stdout and stderr during execution."""
    try:
        with open(os.devnull, "w") as devnull:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = devnull, devnull
            yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
