import { Outlet, useNavigate, useLocation } from "react-router-dom";
import { Layout as AntLayout, Menu, Typography } from "antd";
import { MessageOutlined, SearchOutlined, FileTextOutlined, ExperimentOutlined } from "@ant-design/icons";

const { Header, Content } = AntLayout;
const { Title } = Typography;

export default function Layout() {
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems = [
    {
      key: "/",
      icon: <MessageOutlined />,
      label: "Chat",
    },
    {
      key: "/search",
      icon: <SearchOutlined />,
      label: "Search",
    },
    {
      key: "/retrieval-test",
      icon: <ExperimentOutlined />,
      label: "Retrieval Test",
    },
    {
      key: "/parsing",
      icon: <FileTextOutlined />,
      label: "Parsing",
    },
  ];

  const handleMenuClick = (e: { key: string }) => {
    navigate(e.key);
  };

  return (
    <AntLayout style={{ minHeight: "100vh" }}>
      <Header style={{ display: "flex", alignItems: "center", padding: "0 24px" }}>
        <Title level={4} style={{ color: "white", margin: "0 24px 0 0" }}>
          PE Agent
        </Title>
        <Menu
          theme="dark"
          mode="horizontal"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={handleMenuClick}
          style={{ flex: 1, minWidth: 0 }}
        />
      </Header>
      <Content style={{ padding: "0" }}>
        <Outlet />
      </Content>
    </AntLayout>
  );
}