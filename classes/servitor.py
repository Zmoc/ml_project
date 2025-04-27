import contextlib
import os
import sys
import nmap
from scapy.all import IP, TCP, wrpcap
import random
import pyshark
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

        self.role_frame = (
            "You are a cybersecurity agent tasked with defending the local system."
        )
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
            "My name is Servitor. I am a cybersecurity agent tasked with defending your local system."
        )

    def get_input(self):
        print("Servitor is ready. Type your command:")

        # Dictionary to map intents to functions
        actions = {
            "scan_ports": self.scan_ports,
            "simulate_ddos": self.sim_ddos,
            "scan_traffic": self.scan_traffic,
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
                print("Agent exiting.")
                break

            intent = self.intent(user_input)

            try:
                # Use the dictionary to get the appropriate action
                action = actions.get(intent, self.handle_unrecognized)
                action()
            except Exception as e:
                print(f"Error: {e}")

    def scan_ports(self):
        scan_result, scan_prompt = self.port_scan()
        try:
            print("[*] Generating summary...")
            summary = self.summarize_report(scan_result, scan_prompt)
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
            print("[*] Generating summary...")
            summary = self.summarize_report(scan_result, scan_prompt)
            print("\nSummary:\n", summary)
        except Exception as e:
            print(f"[!] Failed to summarize report: {e}")

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
            arguments="-p 20,21,22,23,25,53,80,443,110,143,161,162,3306,3389,445,5900,8080,5060"
            " -sV",
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

        report_type_prompt = (
            "These are the results of an NMAP port scan. "
            "They describe the status of all ports on the system and what services are currently running."
        )
        return scan_result, report_type_prompt

    def sim_syn_flood(self, target_ip, target_port, num_packets, output_pcap):
        packets = []  # List to store the generated packets

        for _ in range(num_packets):
            fake_src_ip = ".".join(
                str(random.randint(1, 254)) for _ in range(4)
            )  # Generate random fake source IP
            ip = IP(src=fake_src_ip, dst=target_ip)  # Randomize source IP
            syn = TCP(
                sport=random.randint(1024, 65535),
                dport=int(target_port),
                flags="S",
                seq=random.randint(1, 100000),
            )  # Random sport and seq
            packet = ip / syn
            packets.append(packet)  # Append the packet to the list

        # Save the packets to a pcap file
        wrpcap(output_pcap, packets)
        print(
            f"Simulated {num_packets} SYN packets from random IPs to {target_ip}:{target_port} and saved to {output_pcap}"
        )

    def traffic_scan(self):
        traffic_results = self.evaluate_traffic("data/shark/log.pcap")
        report_type_prompt = (
            "These are the results of a Wireshark pcap scan. "
            "They describe the number of packets that were captured and other information "
            "about them along with suspicion scores for SYN flood and a DDos attack."
        )
        return traffic_results, report_type_prompt

    def evaluate_traffic(self, pcap_file):
        from collections import Counter
        import pyshark
        import nest_asyncio

        nest_asyncio.apply()

        cap = pyshark.FileCapture(pcap_file, display_filter="tcp")

        total_packets = 0
        syn_count = 0
        syn_packets = []
        source_ips = Counter()
        target_ips = Counter()

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
                continue  # Skip packets missing IP/TCP fields

        cap.close()

        # Prepare output as a single string
        output = ""

        # --- SYN Flood Detection ---
        output += "\n--- SYN Flood Evaluation ---\n"
        output += f"Total packets: {total_packets}\n"
        output += f"SYN packets without ACK: {syn_count}\n"

        syn_ratio = (syn_count / total_packets) * 100 if total_packets > 0 else 0
        output += f"SYN flood suspicion score: {syn_ratio:.2f}%\n"

        if syn_ratio > 50:
            output += "⚠️ High chance of SYN flood detected.\n"
        else:
            output += "✅ Traffic looks normal.\n"

        # --- DDoS Detection ---
        output += "\n--- DDoS Evaluation ---\n"
        output += f"Unique attacking IPs: {len(source_ips)}\n"
        output += f"Most targeted IP: {target_ips.most_common(1)}\n"

        if len(source_ips) > 10 and target_ips.most_common(1)[0][1] > 50:
            output += (
                "🚨 DDoS likely detected: multiple sources targeting one destination.\n"
            )
        else:
            output += "✅ No strong evidence of DDoS attack.\n"

        return output

    def query(self, prompt, max_tokens=512):
        print("Prompt Recieved.")
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            echo=False,
            stream=False,
            temperature=0.7,
            top_p=0.95,
        )

        print("Prompt processed.")

        # Print raw response to inspect
        print(f"Raw response: {response}")

        # Clean up the response
        response_text = response["choices"][0]["text"].strip()
        response_text = "\n".join(
            [line.strip() for line in response_text.splitlines() if line.strip()]
        )  # Remove empty lines

        if len(response_text) == 0:  # If response is still empty, handle gracefully
            response_text = "No meaningful response generated."

        return response_text

    def intent(self, user_input, prefix="", suffix=""):
        """Determines the user's intent based on input."""
        framed_input = f"{prefix.strip()} {user_input.strip()} {suffix.strip()}".strip()
        intent_prompt = f"""[INST]

{self.role_frame}

Decide what the following instruction is asking you to do.

Instruction: "{framed_input}"

Respond with only one of the following:
    scan_ports
    simulate_ddos
    scan_traffic
    ignore
    request_clarification

    Only use these keywords exactly.[/INST]
"""
        print("Processing prompt")
        llm_response = self.query(intent_prompt)
        print(f"Response: {llm_response}")

        # Normalize the response by removing escape characters and converting to lower case
        response_normalized = re.sub(
            r"\\_", "_", llm_response.strip().lower()
        )  # Replace escaped underscores

        print(f"Normalized Response: {response_normalized}")  # Debugging line

        # Map the normalized response to the appropriate action
        return {
            "scan_ports": "scan_ports",
            "simulate_ddos": "simulate_ddos",
            "scan_traffic": "scan_traffic",
            "ignore": "ignore",
            "request_clarification": "request_clarification",
        }.get(response_normalized, "unrecognized")

    def summarize_report(self, report_text, report_prompt):
        """Asks the LLM to summarize the port scan report."""
        summary_prompt = f"""[INST]
    {self.role_frame}
    {report_prompt}
    {report_text}

    Please summarize this report and highlight any items that might require a security review. Be specific and give actionable steps on how to address the issues.[/INST]
    """
        return self.query(summary_prompt, max_tokens=512)


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
