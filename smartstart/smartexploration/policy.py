import numpy as np


class PolicyNode(object):
    """ """
    new_id = 0

    def __init__(self, state, action=None, parent=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self._id = PolicyNode.new_id
        PolicyNode.new_id += 1
        if parent is not None:
            parent.add_child(self)

    def set_children(self, children):
        """

        Parameters
        ----------
        children :
            

        Returns
        -------

        """
        self.children = children
        for child in children:
            child.parent = self

    def add_child(self, child):
        """

        Parameters
        ----------
        child :
            

        Returns
        -------

        """
        idx = self.__len__()
        self.children.append(child)
        return idx

    def get_policy(self):
        """ """
        if self.parent is not None:
            parent_policy = self.parent.get_policy()
            policy = parent_policy + [self.action]
        else:
            policy = []
        return policy

    def __len__(self):
        return len(self.get_policy())

    def __repr__(self):
        children = [child._id for child in self.children]
        parent = str(self.parent._id) if self.parent is not None else None
        return "%s(id=%d, state=%s, length=%d, parent=%s, children=%s)" % \
               (__class__.__name__, self._id, str(self.state), self.__len__(), parent, children)


class PolicyMap(object):
    """ """

    def __init__(self, init_state):
        self.nodes = {}
        self.root = PolicyNode(init_state)
        self.nodes[str(init_state)] = self.root
        self.current_node = self.root

    def reset_to_root(self):
        """ """
        self.current_node = self.root

    def get_node(self, state):
        """

        Parameters
        ----------
        state :
            

        Returns
        -------

        """
        node = None
        try:
            node = self.nodes[str(state)]
        except KeyError:
            pass
        finally:
            return node

    def add_node(self, next_state, action):
        """

        Parameters
        ----------
        next_state :
            
        action :
            

        Returns
        -------

        """
        node = self.get_node(next_state)
        new_node = PolicyNode(next_state, action, self.current_node)
        if node is None:
            self.nodes[str(next_state)] = new_node
        elif len(new_node) <= len(node):
            self.update_node(new_node)
        else:
            new_node = node
        self.current_node = new_node
        return self.current_node

    def update_node(self, new_node):
        """

        Parameters
        ----------
        new_node :
            

        Returns
        -------

        """
        current_node = self.nodes[str(new_node.state)]
        new_node.set_children(current_node.children)
        self.nodes[str(new_node.state)] = new_node

    def __len__(self):
        return len(self.nodes)

    def __repr__(self):
        return "%s(num_nodes=%d)" % (__class__.__name__, len(self.nodes))

    def __str__(self):
        output_str = ""
        ids = []
        for node in self.nodes.values():
            ids.append(node._id)
        for id, node in sorted(zip(ids, self.nodes.values())):
            output_str += str(node) + "\n"
        return output_str
