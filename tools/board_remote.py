"""SSH helper: credentials in environment/SSH agent, never in project files."""
import argparse
import os
from pathlib import Path
import sys
import paramiko

def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='replace')
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host',required=True);ap.add_argument('--user',default='seeed')
    ap.add_argument('--command');ap.add_argument('--upload',nargs=2);ap.add_argument('--download',nargs=2)
    args=ap.parse_args()
    client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(args.host,username=args.user,password=os.environ.get('BOARD_SSH_PASSWORD'),timeout=15)
        if args.upload:
            with client.open_sftp() as s:s.put(*args.upload)
        status=0
        if args.command:
            _,out,err=client.exec_command(args.command)
            print(out.read().decode(errors='replace'),end='')
            print(err.read().decode(errors='replace'),end='',file=sys.stderr)
            status=out.channel.recv_exit_status()
        if args.download and status==0:
            Path(args.download[1]).parent.mkdir(parents=True,exist_ok=True)
            with client.open_sftp() as s:s.get(*args.download)
        raise SystemExit(status)
    finally:client.close()

if __name__=='__main__':main()
