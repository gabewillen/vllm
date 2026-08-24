#!/bin/sh
# cliproxyapi egress: host only on vLLM :8012, LAN only DNS at the router, WAN open.
NET=172.30.0.0/24
DNS=192.168.2.1
for c in INPUT DOCKER-USER; do iptables -S $c | grep -- "-s $NET" | sed 's/^-A/-D/' | while read -r r; do iptables $r; done; done
# host (INPUT): established replies, vLLM, drop everything else
iptables -I INPUT 1 -s $NET -j DROP
iptables -I INPUT 1 -s $NET -p tcp --dport 8012 -j ACCEPT
iptables -I INPUT 1 -s $NET -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
# LAN (FORWARD via DOCKER-USER): DNS to router only, drop other private ranges
iptables -I DOCKER-USER 1 -s $NET -d 10.0.0.0/8 -j DROP
iptables -I DOCKER-USER 1 -s $NET -d 172.16.0.0/12 -j DROP
iptables -I DOCKER-USER 1 -s $NET -d 192.168.0.0/16 -j DROP
iptables -I DOCKER-USER 1 -s $NET -d $DNS -p udp --dport 53 -j ACCEPT
iptables -I DOCKER-USER 1 -s $NET -d $DNS -p tcp --dport 53 -j ACCEPT
