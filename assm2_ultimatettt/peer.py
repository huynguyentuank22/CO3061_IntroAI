import socket
import threading
import time
import pickle
from typing import List, Dict
import logging
import requests


class PeerNetwork:
    def __init__(self, username: str, game=None, on_game_start=None, on_opponent_move=None, on_pause_update=None, on_disconnect=None):
        self.username = username
        self.game = game  # Optional legacy game instance
        # Optional callbacks to integrate with Pygame game
        self.on_game_start = on_game_start
        self.on_opponent_move = on_opponent_move
        self.on_pause_update = on_pause_update
        self.on_disconnect = on_disconnect
        self.local_ip = self.get_local_ip()
        self.UDP_PORT = 5005
        self.udp_socket = None
        self.tcp_socket = None
        self.tcp_port = None
        self.peer_connection = None
        self.is_connected = False
        self.pending_requests: List[Dict] = []
        self.is_broadcasting = False
        self.broadcast_thread = None
        self.request_lock = threading.Lock()
        self.opponent_username = None
        self.game_status = None  # To store game status messages
        self.ready = False  # My ready status
        self.opponent_ready = False  # Opponent's ready status
        self.accepted_connection = False
        # Pause/Resume state
        self.pause_active = False
        self.pause_deadline_ts = None
        self.my_pause_count = 0
        self.opponent_pause_count = 0
        self.my_ack = False
        self.opponent_ack = False

    def get_local_ip(self):
        """Get local IP address."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 80))
            local_ip = s.getsockname()[0]
            print(f"Local IP: {local_ip}")
        except Exception:
            local_ip = '127.0.0.1'
            print("Failed to get local IP, using localhost")
        finally:
            s.close()
        return local_ip

    def initialize_udp_socket(self):
        """Initialize UDP socket for broadcasting and listening."""
        try:
            # Close existing socket if any
            if self.udp_socket:
                try:
                    self.udp_socket.close()
                except:
                    pass
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Set socket to non-blocking for better control
            self.udp_socket.settimeout(0.1)  # 100ms timeout for recv
            # Bind to all interfaces for better broadcast reception
            self.udp_socket.bind(('', self.UDP_PORT))
            print(f"UDP socket initialized on port {self.UDP_PORT}")
            return True
        except Exception as e:
            print(f"UDP socket initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def initialize_tcp_socket(self):
        """Initialize TCP socket for direct communication."""
        try:
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.tcp_socket.bind(('0.0.0.0', 0))
            self.tcp_port = self.tcp_socket.getsockname()[1]
            self.tcp_socket.listen(1)
            print(f"TCP socket initialized on port {self.tcp_port}")
            
            # Start TCP listener thread
            tcp_listener_thread = threading.Thread(target=self.listen_for_tcp, daemon=True)
            tcp_listener_thread.start()
            
            return True
        except Exception as e:
            print(f'Initialize TCP failed: {e}')
            return False

    def listen_for_tcp(self):
        """Listen for incoming TCP connections."""
        print(f"TCP listener started for {self.username} on port {self.tcp_port}")
        while True:
            try:
                if not self.tcp_socket:
                    break
                client_socket, client_address = self.tcp_socket.accept()
                print(f"TCP connection attempt from {client_address}, is_connected={self.is_connected}")
                if not self.is_connected:
                    self.peer_connection = client_socket
                    self.is_connected = True
                    print(f"Accepted TCP connection from {client_address}")
                    
                    # Start message handling thread
                    threading.Thread(target=self.handle_peer_messages, 
                                   daemon=True).start()
                    
                    # Wait a moment for message handler to start
                    time.sleep(0.1)
                    
                    # Send connection confirmation
                    self.send_message({
                        'type': 'CONNECTION_ACCEPTED',
                        'username': self.username
                    })
                    print(f"Sent CONNECTION_ACCEPTED to {client_address}")
                else:
                    # Reject connection if already connected
                    print(f"Rejecting connection from {client_address} - already connected")
                    client_socket.close()
            except socket.error as e:
                if self.tcp_socket:
                    print(f"TCP accept error: {e}")
                break
            except Exception as e:
                print(f"TCP listener error: {e}")
                import traceback
                traceback.print_exc()
                if not self.tcp_socket:
                    break

    def start_udp_listener(self):
        """Start the UDP listener thread (non-blocking)."""
        if not self.udp_socket:
            if not self.initialize_udp_socket():
                return False
        # Start UDP listener thread if not already running
        # Check if thread is already running by checking if socket is bound
        udp_listener_thread = threading.Thread(target=self.listen_for_udp, daemon=True)
        udp_listener_thread.start()
        print(f"UDP listener thread started for {self.username}")
        return True

    def start(self):
        """Start the peer network (legacy console mode)."""
        self.initialize_udp_socket()
        self.start_udp_listener()

        while True:
            if not self.is_connected:
                choice = input(
                    "\nDo you want to:\n1. Broadcast connection request\n2. Just listen\n3. Quit\nChoice: ")

                if choice == '1':
                    self.broadcast_connect_request()
                    time.sleep(10)
                elif choice == '2':
                    print("Continuing to listen for requests...")
                    time.sleep(10)
                elif choice == '3':
                    break
            else:
                message = input("Enter message (or 'quit' to disconnect): ")
                if message.lower() == 'quit':
                    self.is_connected = False
                    if self.peer_connection:
                        self.peer_connection.close()
                else:
                    self.send_message(message)

    def broadcast_connect_request(self):
        """Start broadcasting connection requests with improved reliability."""
        if not self.tcp_socket:
            if not self.initialize_tcp_socket():
                return False

        if not self.udp_socket:
            if not self.initialize_udp_socket():
                return False

        print(f"Broadcasting connection request from {self.username}...")
        self.is_broadcasting = True
        
        def broadcast_loop():
            broadcast_count = 0
            while self.is_broadcasting and not self.is_connected:
                try:
                    request_msg = pickle.dumps({
                        'type': 'CONNECT_REQUEST',
                        'username': self.username,
                        'local_ip': self.local_ip,
                        'tcp_port': self.tcp_port,
                        'sequence': broadcast_count
                    })
                    
                    # Send to multiple broadcast addresses for better compatibility
                    # Try subnet broadcast first (most reliable on LAN)
                    try:
                        subnet_broadcast = '.'.join(self.local_ip.split('.')[:-1] + ['255'])
                        self.udp_socket.sendto(request_msg, (subnet_broadcast, self.UDP_PORT))
                    except:
                        pass
                    
                    # Also try global broadcast
                    try:
                        self.udp_socket.sendto(request_msg, ('255.255.255.255', self.UDP_PORT))
                    except:
                        pass
                    
                    broadcast_count += 1
                    time.sleep(1)  # Broadcast more frequently
                except Exception as e:
                    print(f"Broadcasting error: {e}")
                    time.sleep(1)  # Prevent tight loop on error
                    continue

        self.broadcast_thread = threading.Thread(target=broadcast_loop, daemon=True)
        self.broadcast_thread.start()
        return True

    def stop_broadcasting(self):
        """Stop broadcasting connection requests."""
        self.is_broadcasting = False
        if self.broadcast_thread:
            self.broadcast_thread.join(timeout=1)

    def listen_for_udp(self):
        """Listen for incoming UDP messages with improved error handling and validation."""
        print(f"UDP listener thread started for {self.username}")
        while True:
            try:
                if not self.udp_socket:
                    time.sleep(1)
                    continue
                data, addr = self.udp_socket.recvfrom(4096)
                if not data:
                    continue

                try:
                    message = pickle.loads(data)
                except (pickle.UnpicklingError, EOFError) as e:
                    continue  # Silently skip invalid messages

                if not isinstance(message, dict) or 'type' not in message:
                    continue

                if (message['type'] == 'CONNECT_REQUEST' and 
                    message.get('username') != self.username and  # Ignore self-broadcasts
                    not self.is_connected):  # Only process if not already connected
                    
                    # Validate required fields
                    required_fields = ['username', 'local_ip', 'tcp_port']
                    if not all(field in message for field in required_fields):
                        continue

                    # Use the IP from the message or the sender address
                    sender_ip = addr[0]
                    request = {
                        'username': message['username'],
                        'ip': sender_ip,  # Use actual sender IP
                        'tcp_port': message['tcp_port'],
                        'timestamp': time.time(),
                        'strength': 1
                    }

                    self.update_pending_requests(request)
                    print(f"Received connection request from {request['username']} at {request['ip']}:{request['tcp_port']}")

            except socket.timeout:
                # Expected with non-blocking socket, continue
                continue
            except socket.error as e:
                # Socket might be closed, check if we should continue
                if self.udp_socket:
                    time.sleep(0.1)
                else:
                    break
            except Exception as e:
                print(f"Unexpected error in UDP listener: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def update_pending_requests(self, new_request: Dict):
        """Update pending requests with thread safety and improved handling."""
        with self.request_lock:
            current_time = time.time()
            
            # Remove expired requests (older than 30 seconds)
            self.pending_requests = [
                r for r in self.pending_requests 
                if current_time - r['timestamp'] < 30
            ]
            
            # Check if request from this user already exists
            existing_request = next(
                (r for r in self.pending_requests 
                 if r['username'] == new_request['username']),
                None
            )
            
            if existing_request:
                # Update existing request
                existing_request['timestamp'] = current_time
                existing_request['strength'] += 1
                existing_request['ip'] = new_request['ip']
                existing_request['tcp_port'] = new_request['tcp_port']
            else:
                # Add new request
                self.pending_requests.append(new_request)

    def display_pending_requests(self):
        """Display current pending requests in a formatted way."""
        with self.request_lock:
            if not self.pending_requests:
                print("\nNo pending connection requests.")
                return

            print("\nPending connection requests:")
            print("-" * 50)
            for i, request in enumerate(self.pending_requests, 1):
                age = time.time() - request['timestamp']
                print(f"{i}. Username: {request['username']}")
                print(f"   IP: {request['ip']}:{request['tcp_port']}")
                print(f"   Age: {age:.1f} seconds")
                print(f"   Signal Strength: {'█' * request['strength']}")
                print("-" * 50)

    def get_pending_requests(self):
        """Get list of pending connection requests."""
        print(f"Getting pending requests for {self.username}")
        with self.request_lock:
            current_time = time.time()
            # Clean up old requests before returning
            self.pending_requests = [
                r for r in self.pending_requests 
                if current_time - r['timestamp'] < 30
            ]
            # Filter out self requests
            filtered_requests = [
                r for r in self.pending_requests 
                if r['username'] != self.username
            ]
            print(f"Current pending requests: {self.pending_requests}")
            # Return only necessary information for the frontend
            return [{
                'username': r['username'],
                'timestamp': r['timestamp'],
                'strength': r['strength']
            } for r in filtered_requests]

    def accept_connection(self, opponent_username):
        """Accept a connection request from a specific user."""
        if self.is_connected:
            print(f"Already connected, cannot accept connection from {opponent_username}")
            return False
            
        with self.request_lock:
            request = None
            for req in self.pending_requests:
                if req['username'] == opponent_username:
                    request = req
                    break
        
        if not request:
            print(f"No pending request found for {opponent_username}")
            return False
            
        try:
            # Stop broadcasting if we're searching
            self.stop_broadcasting()
            
            # Connect to peer (they are listening on their TCP port)
            peer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print(f"Attempting to connect to {request['ip']}:{request['tcp_port']}")
            # Set a timeout for the connection attempt
            peer_socket.settimeout(5)
            try:
                peer_socket.connect((request['ip'], request['tcp_port']))
                print(f"Successfully connected to {request['ip']}:{request['tcp_port']}")
            except socket.timeout:
                print(f"Connection timeout to {request['ip']}:{request['tcp_port']}")
                peer_socket.close()
                return False
            except Exception as e:
                print(f"Connection failed: {e}")
                import traceback
                traceback.print_exc()
                peer_socket.close()
                return False
            
            # Reset to blocking mode after connection
            peer_socket.settimeout(None)
            self.peer_connection = peer_socket
            self.is_connected = True
            self.opponent_username = opponent_username
            print(f"Connected to peer {opponent_username} at {request['ip']}:{request['tcp_port']}")

            # Start message handling thread
            threading.Thread(target=self.handle_peer_messages,
                          daemon=True).start()
            
            # Clean up requests
            with self.request_lock:
                self.pending_requests = []
            
            # Wait a moment for message handler to start
            time.sleep(0.15)
            
            # Send connection confirmation
            self.send_message({
                'type': 'CONNECTION_ACCEPTED',
                'username': self.username
            })
            print(f"Sent CONNECTION_ACCEPTED to {opponent_username}")
            
            # Wait a moment for CONNECTION_ACCEPTED to be received by the other side
            # The broadcaster will send GAME_START after receiving CONNECTION_ACCEPTED
            # So we don't send GAME_START here - we wait for it from the broadcaster
            self.accepted_connection = False  # We are the acceptor, not the one who accepted
            
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            import traceback
            traceback.print_exc()
            # Clean up failed connection
            try:
                if 'peer_socket' in locals():
                    peer_socket.close()
            except:
                pass
            self.is_connected = False
            return False

    def reject_connection(self, username):
        """Reject a connection request from a specific user."""
        self.pending_requests = [
            r for r in self.pending_requests 
            if r['username'] != username
        ]

    def handle_peer_messages(self):
        """Handle incoming messages from connected peer."""
        print(f"Message handler thread started for {self.username}")
        while self.is_connected and self.peer_connection:
            try:
                # First receive the message length (4 bytes)
                length_data = b''
                while len(length_data) < 4:
                    chunk = self.peer_connection.recv(4 - len(length_data))
                    if not chunk:
                        print(f"Connection closed while receiving message length")
                        self.handle_disconnect("Opponent disconnected")
                        break
                    length_data += chunk
                
                if not length_data or len(length_data) < 4:
                    break
                    
                message_len = int.from_bytes(length_data, 'big')
                if message_len > 1024 * 1024:  # Sanity check: max 1MB
                    print(f"Invalid message length: {message_len}")
                    self.handle_disconnect("Invalid message received")
                    break
                
                # Now receive the actual message
                message_data = b''
                while len(message_data) < message_len:
                    chunk = self.peer_connection.recv(min(4096, message_len - len(message_data)))
                    if not chunk:
                        print(f"Connection closed while receiving message data")
                        self.handle_disconnect("Opponent disconnected")
                        break
                    message_data += chunk
                
                if len(message_data) < message_len:
                    break
                
                try:
                    message = pickle.loads(message_data)
                    print(f"Received message type: {message.get('type')}")
                except (pickle.UnpicklingError, EOFError) as e:
                    print(f"Failed to unpickle message: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                if message.get('type') == 'MOVE':
                    print(f"Received move: {message}")
                    if self.on_opponent_move:
                        self.on_opponent_move(message['main_row'], message['main_col'], message['sub_row'], message['sub_col'])
                    elif self.game:
                        # Legacy behavior with console game
                        result = self.game.receive_move(
                            message['main_row'],
                            message['main_col'],
                            message['sub_row'],
                            message['sub_col']
                        )
                        self.game.print_board()
                        self.game_status = {
                            'type': 'MOVE',
                            'main_row': message['main_row'],
                            'main_col': message['main_col'],
                            'sub_row': message['sub_row'],
                            'sub_col': message['sub_col'],
                            'sub_board_result': result.get('sub_board_result'),
                            'game_over': result.get('game_over'),
                            'winner': result.get('winner'),
                            'is_draw': result.get('is_draw')
                        }
                    else:
                        # Network mode: store move in game_status for main loop to process
                        self.game_status = {
                            'type': 'MOVE',
                            'main_row': message['main_row'],
                            'main_col': message['main_col'],
                            'sub_row': message['sub_row'],
                            'sub_col': message['sub_col']
                        }
                        print(f"Stored move in game_status for processing")
                elif message.get('type') == 'GAME_START':
                    first_player = message.get('first_player')
                    print(f"Received GAME_START, first player: {first_player}")
                    # Store in game_status so main loop can detect it
                    self.game_status = {
                        'type': 'GAME_START',
                        'first_player': first_player
                    }
                    if self.on_game_start:
                        self.on_game_start(first_player)
                elif message.get('type') == 'CONNECTION_ACCEPTED':
                    # store opponent username
                    self.opponent_username = message.get('username')
                    print(f"Received CONNECTION_ACCEPTED from {self.opponent_username}")
                    # If we were broadcasting (TCP listener accepted connection), send GAME_START
                    # The acceptor connects to us, so we send GAME_START
                    if self.opponent_username:
                        time.sleep(0.2)  # Brief delay to ensure connection is stable
                        first_player = min(self.username, self.opponent_username)
                        game_start_msg = {
                            'type': 'GAME_START',
                            'first_player': first_player
                        }
                        self.send_message(game_start_msg)
                        print(f"Sent GAME_START, first player: {first_player}")
                        # Store in game_status so main loop can detect it
                        self.game_status = {
                            'type': 'GAME_START',
                            'first_player': first_player
                        }
                        print(f"Stored GAME_START in game_status: {self.game_status}")
                elif message.get('type') == 'DISCONNECT':
                    self.handle_disconnect(message.get('message', 'Opponent disconnected'))
                    break
                elif message.get('type') == 'PAUSE_REQUEST':
                    # Opponent requests pause
                    self.pause_active = True
                    self.pause_deadline_ts = time.time() + message.get('duration', 30)
                    self.opponent_pause_count = message.get('opponent_pause_count', self.opponent_pause_count)
                    self.my_ack = False
                    self.opponent_ack = False
                    if self.on_pause_update:
                        self.on_pause_update({
                            'pause_active': True,
                            'deadline_ts': self.pause_deadline_ts,
                            'my_ack': self.my_ack,
                            'opponent_ack': self.opponent_ack,
                            'my_pause_count': self.my_pause_count,
                            'opponent_pause_count': self.opponent_pause_count
                        })
                elif message.get('type') == 'PAUSE_ACK':
                    # Opponent pressed Continue
                    self.opponent_ack = True
                    if self.on_pause_update:
                        self.on_pause_update({
                            'pause_active': self.pause_active,
                            'deadline_ts': self.pause_deadline_ts,
                            'my_ack': self.my_ack,
                            'opponent_ack': self.opponent_ack,
                            'my_pause_count': self.my_pause_count,
                            'opponent_pause_count': self.opponent_pause_count
                        })
                    # Check if both are ready or time expired, and try to resume
                    self.try_resume()
                elif message.get('type') == 'RESUME':
                    # Resume game - opponent sent resume signal
                    print(f"[PAUSE] Received RESUME message from opponent")
                    self.pause_active = False
                    self.pause_deadline_ts = None
                    self.my_ack = False
                    self.opponent_ack = False
                    if self.on_pause_update:
                        self.on_pause_update({'pause_active': False})
                elif message.get('type') == 'QUIT':
                    self.handle_disconnect("Opponent quit the game")
                    break
                else:
                    print(f"Received unknown message type: {message}")
            except (ConnectionResetError, BrokenPipeError, OSError) as e:
                print(f"Connection error: {e}")
                self.handle_disconnect("Connection error occurred")
                break
            except Exception as e:
                print(f"Message handling error: {e}")
                import traceback
                traceback.print_exc()
                # Don't disconnect on unexpected errors, just log and continue
                time.sleep(0.1)
                continue

    def handle_disconnect(self, reason="Connection lost"):
        """Handle disconnection with cleanup."""
        self.is_connected = False
        self.game_status = reason
        if self.peer_connection:
            try:
                self.peer_connection.close()
            except:
                pass
        self.peer_connection = None
        self.opponent_username = None
        print(f"Peer connection lost: {reason}")
        if self.on_disconnect:
            self.on_disconnect(reason)

    def send_message(self, message):
        """Send message to connected peer."""
        if self.is_connected and self.peer_connection:
            try:
                serialized_message = pickle.dumps(message)
                # Send message length first (4 bytes)
                message_len = len(serialized_message)
                self.peer_connection.sendall(message_len.to_bytes(4, 'big'))
                # Then send the actual message
                self.peer_connection.sendall(serialized_message)
                print(f"Sent: {message.get('type', 'unknown')} ({message_len} bytes)")
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                print(f"Message send error (connection lost): {e}")
                self.handle_disconnect("Connection lost while sending")
            except Exception as e:
                print(f"Message send error: {e}")
                import traceback
                traceback.print_exc()
                self.is_connected = False

    def get_game_status(self):
        """Get current game status."""
        status = self.game_status
        # Don't clear GAME_START immediately - keep it until game actually starts
        # Other status types can be cleared
        if status and isinstance(status, dict) and status.get('type') == 'GAME_START':
            # Keep GAME_START until it's consumed by starting the game
            return status
        self.game_status = None  # Clear other status types after reading
        return status

    # High-level helpers for Pygame integration
    def send_move(self, main_row, main_col, sub_row, sub_col):
        self.send_message({
            'type': 'MOVE',
            'main_row': main_row,
            'main_col': main_col,
            'sub_row': sub_row,
            'sub_col': sub_col
        })

    def request_pause(self, duration_seconds=30):
        if self.my_pause_count >= 2:
            return False
        self.my_pause_count += 1
        self.pause_active = True
        self.pause_deadline_ts = time.time() + duration_seconds
        self.my_ack = False
        self.opponent_ack = False
        self.send_message({
            'type': 'PAUSE_REQUEST',
            'duration': duration_seconds,
            'opponent_pause_count': self.my_pause_count  # from opponent perspective
        })
        if self.on_pause_update:
            self.on_pause_update({
                'pause_active': True,
                'deadline_ts': self.pause_deadline_ts,
                'my_ack': self.my_ack,
                'opponent_ack': self.opponent_ack,
                'my_pause_count': self.my_pause_count,
                'opponent_pause_count': self.opponent_pause_count
            })
        return True

    def send_pause_ack(self):
        if not self.pause_active:
            return
        self.my_ack = True
        self.send_message({'type': 'PAUSE_ACK'})
        if self.on_pause_update:
            self.on_pause_update({
                'pause_active': self.pause_active,
                'deadline_ts': self.pause_deadline_ts,
                'my_ack': self.my_ack,
                'opponent_ack': self.opponent_ack,
                'my_pause_count': self.my_pause_count,
                'opponent_pause_count': self.opponent_pause_count
            })
        # Check if both are ready or time expired, and try to resume
        self.try_resume()

    def try_resume(self):
        # Resume if both ack or time expired
        if not self.pause_active:
            return False
        now = time.time()
        # Check if both players are ready OR timer has expired
        both_ready = self.my_ack and self.opponent_ack
        timer_expired = self.pause_deadline_ts and now >= self.pause_deadline_ts
        should_resume = both_ready or timer_expired
        
        if should_resume:
            print(f"[PAUSE] try_resume: both_ready={both_ready}, timer_expired={timer_expired}, should_resume={should_resume}")
            self.pause_active = False
            self.pause_deadline_ts = None
            self.my_ack = False
            self.opponent_ack = False
            # Send RESUME message to opponent
            self.send_message({'type': 'RESUME'})
            print(f"[PAUSE] Sent RESUME message to opponent")
            if self.on_pause_update:
                self.on_pause_update({'pause_active': False})
            return True
        return False

    def send_quit(self):
        try:
            self.send_message({'type': 'QUIT'})
        finally:
            self.handle_disconnect("You quit the game")


def main():
    username = input("Enter your username: ")
    peer = PeerNetwork(username)
    peer.start()


if __name__ == '__main__':
    main()
