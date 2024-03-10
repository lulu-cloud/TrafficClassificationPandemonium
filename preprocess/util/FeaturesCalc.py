import os, statistics
from scapy.all import *

class FeaturesCalc():

    def __init__(self, min_window_size=10):
        self.min_window_size = int(min_window_size)
        assert self.min_window_size > 0, "Valore non valido per min_windows_size. Deve essere maggiore di 0."

        self.features_name = ["Avg_syn_flag", "Avg_urg_flag", "Avg_fin_flag", "Avg_ack_flag", "Avg_psh_flag", "Avg_rst_flag", "Avg_DNS_pkt", "Avg_TCP_pkt",
        "Avg_UDP_pkt", "Avg_ICMP_pkt", "Duration_window_flow", "Avg_delta_time", "Min_delta_time", "Max_delta_time", "StDev_delta_time",
        "Avg_pkts_lenght", "Min_pkts_lenght", "Max_pkts_lenght", "StDev_pkts_lenght", "Avg_small_payload_pkt", "Avg_payload", "Min_payload",
        "Max_payload", "StDev_payload", "Avg_DNS_over_TCP", "Num_pkts"]

        self.total_packets = 0
        self.nb_samples = 0

    def compute_features(self, packets_list):
        """主要计算 feature 的函数

        Args:
            packets_list: 多个 packet 组成的 list, 通常使用 scapy 读取一个 session, 返回的类型就是 scapy.plist.PacketList
        """

        def increment_sample_nb(nb):
            self.nb_samples += nb

        def update_received_pkts(nb):
            self.total_packets += nb

        def compute_avg(list_of_values):
            if (len(list_of_values) == 0):
                return 0.0
            else:
                return float(sum(list_of_values) / len(packets_list))

        def compute_min(list_of_values):
            if (len(list_of_values) == 0):
                return 0.0
            else:
                return float(min(list_of_values))

        def compute_max(list_of_values):
            if (len(list_of_values) == 0):
                return 0.0
            else:
                return float(max(list_of_values))

        def compute_stDev(list_of_values):
            if (len(list_of_values) == 0 or len(list_of_values) == 1):
                return 0.0
            else:
                try:
                    stat = statistics.stdev(list_of_values)
                    return float(stat)
                except:
                    return 0.0

        # Data una lista di pkts restituisce una lista della medesima grandezza con in prima posizione la ratio (DNS/Pkt5Layer)
        # Se non ci sono pacchetti DNS o non ci sono pacchetti di livello 5 che non siano DNS, ritorna come primo elemento della lista 0
        #译文：给定一个 pkts 列表，返回一个相同量级的列表，其比率位于第一个位置 （DNS/Pkt5Layer）
        #如果没有 DNS 数据包或不是 DNS 的第 5 级数据包，则返回列表 0 中的第一项
        def DNS_over_TCP_ratio(packets_list):
            total_DNS = float(sum(compute_DNS_packets(packets_list)))
            ratio_list = []
            total_packet_high_level_list = []  # lista di 1 e 0 dove 1 si ha se il pacchetto e' di liv 5 e 0 in tutti gli altri casi
            list_of_pkt_with_TCP = compute_TCP_packets(packets_list)  # Rispetto alla lista ho 1.0 dove c'e tcp e 0.0 dove non c'e
            list_of_paylod_lenght = compute_packet_TCP_payload_size(packets_list, count_packet_without_payload=True)  # Rispetto alla lista ho len(payload) dove c'e tcp con carico, 0.0 dove c'e tcp senza carico, e None dove non c'e tcp
            # Calcolo quanti paccehtti di liv 5 ho nella finestra che non sia DNS
            if (len(packets_list) == len(list_of_pkt_with_TCP) and len(packets_list) == len(list_of_paylod_lenght)):
                for i in range(0, len(packets_list) - 1):
                    # Controllo se il Pkt ha TCP
                    if (list_of_pkt_with_TCP[i] == 1.0):
                        # Controllo se il pkt ha un payload in quanto vuol dire che e' di liv 5
                        if (list_of_paylod_lenght[i] > 0):
                            # Verifico che il pkt che ha il tcp con un payload, non sia DNS
                            if (not packets_list[i].haslayer("DNS")):
                                total_packet_high_level_list.append(1.0)
                            else:
                                total_packet_high_level_list.append(0.0)
                        # Il pkt non ha payload
                        else:
                            total_packet_high_level_list.append(0.0)
                    # Il pkt non ha tcp
                    else:
                        total_packet_high_level_list.append(0.0)
            else:
                print("Errore imprevisto in dnsOverTCPRatio()")
            total_packet_high_level = float(sum(total_packet_high_level_list))
            if (total_packet_high_level != 0):
                ratio_list.append(float(total_DNS / total_packet_high_level))
            else:
                ratio_list.append(0.0)
            i = 1
            # aggiungo tanti 0 quanto da 1 a len(pktList) - 1
            while (i <= len(packets_list) - 1):
                ratio_list.append(0.0)
                i += 1
            return ratio_list

        #Calcola la durata del flusso di pacchetti.计算数据包流的持续时间
        def compute_duration_flow(packets_list):
            duration_time = packets_list[len(packets_list) - 1].time - packets_list[0].time # 最后一个包的时间 - 第一个包的时间
            return float(duration_time)

        # Calcola la grandezza in byte di ogni pacchetto in una lista di pacchetti
        def packets_bytes_lenght(packets_list):
            pkt_lenght_list = []
            for pkt in packets_list:
                pkt_lenght_list.append(float(len(pkt)))
            return pkt_lenght_list

        # Calcola il numero di pacchetti DNS
        def compute_DNS_packets(packets_list):
            dns_counter = []
            for pkt in packets_list:
                if (pkt.haslayer("DNS")):
                    dns_counter.append(1.0)
                else:
                    dns_counter.append(0.0)
            return dns_counter

        # Calcola il numero di pacchetti TCP
        def compute_TCP_packets(packets_list):
            tcp_counter = []
            for pkt in packets_list:
                if (pkt.haslayer("TCP")):
                    tcp_counter.append(1.0)
                else:
                    tcp_counter.append(0.0)
            return tcp_counter

        # Calcola il numero di pacchetti UDP
        def compute_UDP_packets(ackets_list):
            udp_counter = []
            for pkt in packets_list:
                if (pkt.haslayer("UDP")):
                    udp_counter.append(1.0)
                else:
                    udp_counter.append(0.0)
            return udp_counter

        # Calcola il numero di pacchetti ICMP,
        def compute_ICMP_packets(packets_list):
            icmp_counter = []
            for pkt in packets_list:
                if (pkt.haslayer("ICMP") is True):
                    icmp_counter.append(1.0)
                else:
                    icmp_counter.append(0.0)
            return icmp_counter

        # Conta il numero di pacchetti con il layer tcp che hanno payload piccolo o assente
        def compute_packet_with_small_TCP_payload(packets_list, count_packet_without_payload=False):
            packets_small_payload_count = []
            pktPayloadList = compute_packet_TCP_payload_size(packets_list, count_packet_without_payload=count_packet_without_payload)
            for payload in pktPayloadList:
                if (payload <= 32):  # 32 e' stato scelto in base al framework bonesi che simula una botnet e imposta di default il paylaod pare a 32
                    packets_small_payload_count.append(1.0)
                elif (payload > 32):
                    packets_small_payload_count.append(0.0)
                elif (payload == None):
                    # Se ha il layer tcp e non rispetta i canoni aumenta il contatore. Se non ha il layer tcp non incrementa il contatore.
                    # Quindi anche se una finestra e' di 10 pkt, si pesera' questo parametro rispetto il numeri di pkt che hanno il layer TCP
                    if (count_packet_without_payload):
                        packets_small_payload_count.append(0.0)
                    else:
                        pass
            return packets_small_payload_count

        # Calcola la dimensione del payload di un pacchetto TCP
        def compute_packet_TCP_payload_size(packets_list, count_packet_without_payload=False):
            payload_size_list = []
            for pkt in packets_list:
                if (pkt.haslayer("TCP")):
                    if (pkt["TCP"].payload == None):  # Il pacchetto e' TCP ma non ha payload. Probabilemente e' un three way
                        payload_size_list.append(0.0)
                    else:
                        payload_size_list.append(float(len(pkt["TCP"].payload)))
                else:
                    if (count_packet_without_payload):
                        payload_size_list.append(None)
                    else:
                        pass
            return payload_size_list

        def compute_delta_time(packets_list):
            i = 1
            delta_time_list = []
            while (i <= (len(packets_list) - 1)):
                delta_time_list.append(packets_list[i].time - packets_list[i - 1].time)
                i += 1
            return delta_time_list

        # Calcola i flags TCP attivi in un pacchetto. L'array contiene 1 se il flag 计算数据包中的活动 TCP 标志flag。
        # e' attivo, 0 se non lo e' o il pkt non e' TCP    如果处于活动状态，如果不是或 pkt 不是 TCP，则为 0
        def compute_tcp_flags(packets_list):
            syn_counter = []
            fin_counter = []
            ack_counter = []
            psh_counter = []
            urg_counter = []
            rst_counter = []
            FIN = 0x01
            SYN = 0x02
            RST = 0x04
            PSH = 0x08
            ACK = 0x10
            URG = 0x20
            for pkt in packets_list:
                if (pkt.haslayer("TCP")):#haslayer(a)函数：检查参数里的层是否存在
                    F = pkt["TCP"].flags
                    # print(type(F))
                    # print("F是",F)
                    # print("F & FIN是",F & FIN)

                    if F & FIN:
                        fin_counter.append(1.0)
                        syn_counter.append(0.0)
                        ack_counter.append(0.0)
                        psh_counter.append(0.0)
                        urg_counter.append(0.0)
                        rst_counter.append(0.0)
                    elif F & SYN:
                        fin_counter.append(0.0)
                        syn_counter.append(1.0)
                        ack_counter.append(0.0)
                        psh_counter.append(0.0)
                        urg_counter.append(0.0)
                        rst_counter.append(0.0)
                    elif F & RST:
                        fin_counter.append(0.0)
                        syn_counter.append(0.0)
                        ack_counter.append(0.0)
                        psh_counter.append(0.0)
                        urg_counter.append(0.0)
                        rst_counter.append(1.0)
                    elif F & PSH:
                        fin_counter.append(0.0)
                        syn_counter.append(0.0)
                        ack_counter.append(0.0)
                        psh_counter.append(1.0)
                        urg_counter.append(0.0)
                        rst_counter.append(0.0)
                    elif F & ACK:
                        fin_counter.append(0.0)
                        syn_counter.append(0.0)
                        ack_counter.append(1.0)
                        psh_counter.append(0.0)
                        urg_counter.append(0.0)
                        rst_counter.append(0.0)
                    elif F & URG:
                        fin_counter.append(0.0)
                        syn_counter.append(0.0)
                        ack_counter.append(0.0)
                        psh_counter.append(0.0)
                        urg_counter.append(1.0)
                        rst_counter.append(0.0)
                    else:
                        pass
                else:
                    fin_counter.append(0.0)
                    syn_counter.append(0.0)
                    ack_counter.append(0.0)
                    psh_counter.append(0.0)
                    urg_counter.append(0.0)
                    rst_counter.append(0.0)
            return (syn_counter, fin_counter, ack_counter, psh_counter, urg_counter, rst_counter)

        if(len(packets_list) < self.get_min_window_size()):
            # 包数量小于某个数的时候不进行计算, 返回None
            print("\nToo few packets!!!\n")
            return None
        else:
            # 6个标志位的统计量
            syn_lst, fin_lst, ack_lst, psh_lst, urg_lst, rst_lst = compute_tcp_flags(packets_list)
            # print("syn_list是",syn_lst)
            syn_avg = compute_avg(syn_lst)
            # print("syn_avg是", syn_avg)
            fin_avg = compute_avg(fin_lst)
            ack_avg = compute_avg(ack_lst)
            psh_avg = compute_avg(psh_lst)
            urg_avg = compute_avg(urg_lst)
            rst_avg = compute_avg(rst_lst)
            # 一个session的持续时间
            durationFlow = compute_duration_flow(packets_list)
            # 每一个packet之间的间隔时间
            avgTimeFlow = compute_avg(compute_delta_time(packets_list))
            minTimeFlow = compute_min(compute_delta_time(packets_list))
            maxTimeFlow = compute_max(compute_delta_time(packets_list))
            stdevTimeFlow = compute_stDev(compute_delta_time(packets_list))
            # 统计四种协议的包
            dns_pkt = compute_avg(compute_DNS_packets(packets_list))
            tcp_pkt = compute_avg(compute_TCP_packets(packets_list))
            udp_pkt = compute_avg(compute_UDP_packets(packets_list))
            icmp_pkt = compute_avg(compute_ICMP_packets(packets_list))
            # packet的长度
            pktLenghtAvg = compute_avg(packets_bytes_lenght(packets_list))
            pktLenghtMin = compute_min(packets_bytes_lenght(packets_list))
            pktLenghtMax = compute_max(packets_bytes_lenght(packets_list))
            pktLenghtStDev = compute_stDev(packets_bytes_lenght(packets_list))
            # 拥有小载荷的packet的数量
            smallPktPayloadAvg = compute_avg(compute_packet_with_small_TCP_payload(packets_list, False))
            # 计算载荷相关
            avgPayload = compute_avg(compute_packet_TCP_payload_size(packets_list, False))
            minPayload = compute_min(compute_packet_TCP_payload_size(packets_list, False))
            maxPayload = compute_max(compute_packet_TCP_payload_size(packets_list, False))
            stDevPayload = compute_stDev(compute_packet_TCP_payload_size(packets_list, False))
            dnsOverTcpRatioNormalized = compute_avg(DNS_over_TCP_ratio(packets_list))

            # 这个row是最后返回的特征
            row = [syn_avg, urg_avg, fin_avg, ack_avg, psh_avg, rst_avg, dns_pkt, tcp_pkt, udp_pkt, icmp_pkt, durationFlow, avgTimeFlow,
                    minTimeFlow, maxTimeFlow, stdevTimeFlow, pktLenghtAvg, pktLenghtMin, pktLenghtMax, pktLenghtStDev, smallPktPayloadAvg,
                    avgPayload, minPayload, maxPayload, stDevPayload, dnsOverTcpRatioNormalized, len(packets_list)]

            increment_sample_nb(1)
            update_received_pkts(len(packets_list))

            return row

    def get_total_pkts(self):
        return self.total_packets

    def get_total_sample(self):
        return self.nb_samples

    def reset_sample_counter(self):
        self.nb_samples = 0

    def reset_total_pkts_counter(self):
        self.total_packets = 0

    def set_min_window_size(self, val):
        """
        重新设置最小的窗口值
        """
        self.min_window_size = val

    def get_min_window_size(self):
        return self.min_window_size

    def get_features_name(self):
        return self.features_name
    
if __name__ == "__main__":
    # 测试计算 pcap 统计信息
    pcapPath = 'D:\lab\加密流量检测(论文及笔记)\模型训练\Traffic-Classification-master\data\preprocess_data\Chat\\aim_chat_3a\\aim_chat_3a.pcap.TCP_64-12-27-65_443_131-202-240-87_64714.pcap'
    featuresCalc = FeaturesCalc(min_window_size=1) # 初始化
    pkts = rdpcap(pcapPath) # 一整个文件中有多个pcap文件, 读取之后会是一个list
    print(len(pkts))
    print(pkts)
    print(pkts[0].show())
    features = featuresCalc.compute_features(packets_list=pkts) # 计算特征
    print(features)
    print(len(features))
    print('Test End.')