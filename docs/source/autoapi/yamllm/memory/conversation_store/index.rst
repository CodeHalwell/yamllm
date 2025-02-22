yamllm.memory.conversation_store
================================

.. py:module:: yamllm.memory.conversation_store


Classes
-------

.. autoapisummary::

   yamllm.memory.conversation_store.ConversationStore
   yamllm.memory.conversation_store.VectorStore


Module Contents
---------------

.. py:class:: ConversationStore(db_path: str = 'yamllm/memory/conversation_history.db')

   A class to manage conversation history stored in a SQLite database.

   .. attribute:: db_path

      The path to the SQLite database file.

      :type: str

   .. method:: db_exists() -> bool

      
      Check if the database file exists.

   .. method:: create_db() -> None

      
      Create the database and messages table if they don't exist.

   .. method:: add_message(session_id

      str, role: str, content: str) -> int:
      Add a message to the database and return its ID.

   .. method:: get_messages(session_id

      str = None, limit: int = None) -> List[Dict[str, str]]:
      Retrieve messages from the database.

   .. method:: get_session_ids() -> List[str]

      
      Retrieve a list of unique session IDs from the database.

   .. method:: delete_session(session_id

      str) -> None:
      Delete all messages associated with a specific session ID.
      delete_database() -> None:
      Delete the entire database file.

   .. method:: __len__() -> int

      
      Returns the total number of messages in the store.

   .. method:: __str__() -> str

      
      Returns a human-readable string representation.

   .. method:: __repr__() -> str

      
      Returns a detailed string representation.
      


   .. py:method:: add_message(session_id: str, role: str, content: str) -> int

      Add a message to the database and return its ID.

      :param session_id: The ID of the session to which the message belongs.
      :type session_id: str
      :param role: The role of the sender (e.g., 'user', 'assistant').
      :type role: str
      :param content: The content of the message.
      :type content: str

      :returns: The ID of the newly added message in the database.
      :rtype: int



   .. py:method:: create_db() -> None

      Create the database and messages table if they don't exist.

      This method establishes a connection to the SQLite database specified by
      `self.db_path`. It then creates a table named `messages` with the following
      columns if it does not already exist:
          - id: An integer primary key that auto-increments.
          - session_id: A text field that is not null.
          - role: A text field that is not null.
          - content: A text field that is not null.
          - timestamp: A datetime field that defaults to the current timestamp.

      The connection to the database is closed after the table is created.



   .. py:method:: db_exists() -> bool

      Check if the database file exists



   .. py:method:: delete_database() -> None

      Delete the entire database file.



   .. py:method:: delete_session(session_id: str) -> None

      Delete all messages associated with a specific session ID.
      :param session_id: The ID of the session to delete.
      :type session_id: str



   .. py:method:: get_messages(session_id: str = None, limit: int = None) -> List[Dict[str, str]]

      Retrieve messages from the database.
      :param session_id: The session ID to filter messages by. Defaults to None.
      :type session_id: str, optional
      :param limit: The maximum number of messages to retrieve. Defaults to None.
      :type limit: int, optional

      :returns:

                A list of messages, where each message is represented as a dictionary
                                      with 'role' and 'content' keys.
      :rtype: List[Dict[str, str]]



   .. py:method:: get_session_ids() -> List[str]

      Retrieve a list of unique session IDs from the database.
      :returns: A list of unique session IDs.
      :rtype: List[str]



   .. py:attribute:: db_path
      :value: 'yamllm/memory/conversation_history.db'



.. py:class:: VectorStore(vector_dim: int = 1536, store_path: str = 'yamllm/memory/vector_store')

   
   Initializes the ConversationStore object.
   :param vector_dim: The dimensionality of the vectors to be stored. Default is 1536.
   :type vector_dim: int
   :param store_path: The path to the directory where the vector store and metadata will be saved. Default is "yamllm/memory/vector_store".
   :type store_path: str

   .. attribute:: vector_dim

      The dimensionality of the vectors to be stored.

      :type: int

   .. attribute:: store_path

      The path to the directory where the vector store and metadata will be saved.

      :type: str

   .. attribute:: index_path

      The path to the FAISS index file.

      :type: str

   .. attribute:: metadata_path

      The path to the metadata file.

      :type: str

   .. attribute:: index

      The FAISS index for storing vectors.

      :type: faiss.Index

   .. attribute:: metadata

      A list to store message metadata.

      :type: list

   The constructor creates the directory if it doesn't exist, and initializes or loads the FAISS index and metadata.


   .. py:method:: add_vector(vector: List[float], message_id: int, content: str, role: str) -> None

      Add a vector to the index with its associated metadata.

      :param vector: The embedding vector to be added.
      :type vector: List[float]
      :param message_id: Unique identifier for the message.
      :type message_id: int
      :param content: The message content.
      :type content: str
      :param role: The role of the message sender.
      :type role: str

      .. note::

         The vector is L2-normalized before being added to the index.
         Updates are automatically saved to disk.



   .. py:method:: get_vec_and_text() -> Tuple[numpy.ndarray, List[Dict[str, Any]]]

      Retrieve all vectors and their associated metadata.

      :returns:

                A tuple containing:
                    - np.ndarray: Array of shape (n, vector_dim) containing all vectors
                    - List[Dict[str, Any]]: List of metadata dictionaries for each vector
                      with keys: 'id', 'content', 'role'
      :rtype: Tuple[np.ndarray, List[Dict[str, Any]]]

      .. note:: Returns empty array and list if the index is empty.



   .. py:method:: search(query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]

      Search for the k most similar vectors in the index.

      :param query_vector: The query embedding vector.
      :type query_vector: List[float]
      :param k: Number of similar items to return. Defaults to 5.
      :type k: int, optional

      :returns:

                List of dictionaries containing:
                    - id (int): Message ID
                    - content (str): Message content
                    - role (str): Message role
                    - similarity (float): Similarity score
      :rtype: List[Dict[str, Any]]



   .. py:attribute:: index_path


   .. py:attribute:: metadata_path


   .. py:attribute:: store_path
      :value: 'yamllm/memory/vector_store'



   .. py:attribute:: vector_dim
      :value: 1536



